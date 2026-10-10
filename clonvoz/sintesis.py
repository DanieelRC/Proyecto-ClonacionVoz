"""Etapa 4: de los tokens de audio a la voz clonada, por el camino real de XTTS-v2.

XTTS-v2 no decodifica los tokens con el DVAE. Vuelve a pasar la secuencia ya
generada por el GPT-2 para leer sus estados ocultos (los "latentes"), y el
HiFi-GAN convierte esos latentes en la onda, condicionado por un vector de 512
números que describe al hablante. Aquí se llaman esos mismos módulos oficiales.

El DVAE se conserva como rama didáctica: su mel permite ver qué describen los
tokens, y se compara con el mel de la voz que de verdad sale del vocoder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import librosa
import numpy as np

from . import caracteristicas, config
from .generacion import GeneradorOcupado
from .identidad import ModeloNoDisponible, leer_checkpoint


@dataclass
class Sintesis:
    """Resultado de la etapa 4 para una secuencia de tokens."""

    onda: np.ndarray
    sr: int
    latentes: np.ndarray
    embedding: np.ndarray
    mel_salida: np.ndarray
    mel_dvae: np.ndarray | None
    tiempos_ms: dict

    @property
    def duracion_s(self) -> float:
        return len(self.onda) / self.sr

    @property
    def distancia_mel(self) -> float | None:
        """Diferencia media, en log-mel, entre lo que dicen los tokens (DVAE) y la voz final."""
        if self.mel_dvae is None:
            return None
        cuadros = min(self.mel_dvae.shape[1], self.mel_salida.shape[1])
        return float(np.abs(self.mel_dvae[:, :cuadros] - self.mel_salida[:, :cuadros]).mean())

    def como_dict(self) -> dict:
        distancia = self.distancia_mel
        return {
            "sr": self.sr,
            "muestras": int(len(self.onda)),
            "duracion_s": round(self.duracion_s, 3),
            "formas": {"latentes": list(self.latentes.shape), "embedding": list(self.embedding.shape),
                       "mel_salida": list(self.mel_salida.shape),
                       "mel_dvae": None if self.mel_dvae is None else list(self.mel_dvae.shape)},
            "distancia_mel": None if distancia is None else round(distancia, 3),
            "tiempos_ms": {k: round(v, 1) for k, v in self.tiempos_ms.items()},
        }


class Sintetizador:
    """Latentes del GPT-2 y vocoder HiFi-GAN oficiales de XTTS-v2."""

    def __init__(self, carpeta, generador):
        from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
        from TTS.tts.models.xtts import XttsArgs

        datos, pesos = leer_checkpoint(carpeta)
        try:
            campos = XttsArgs.__dataclass_fields__
            a = XttsArgs(**{k: v for k, v in datos["model_args"].items() if k in campos})
            # Mismos argumentos que Xtts.init_models.
            decodificador = HifiDecoder(
                input_sample_rate=a.input_sample_rate, output_sample_rate=a.output_sample_rate,
                output_hop_length=a.output_hop_length, ar_mel_length_compression=a.gpt_code_stride_len,
                decoder_input_dim=a.decoder_input_dim, d_vector_dim=a.d_vector_dim,
                cond_d_vector_in_each_upsampling_layer=a.cond_d_vector_in_each_upsampling_layer)
            prefijo = "hifigan_decoder."
            decodificador.load_state_dict(
                {k[len(prefijo):]: v for k, v in pesos.items() if k.startswith(prefijo)}, strict=True)
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            raise ModeloNoDisponible(
                "No se pudo cargar el HiFi-GAN de XTTS-v2. Comprueba config.json y model.pth. "
                "Detalle: " + str(exc)) from exc
        self._preparar(decodificador, generador)

    @classmethod
    def desde_modulos(cls, decodificador, generador):
        """Construye el sintetizador con módulos ya creados (pruebas y scripts)."""
        objeto = cls.__new__(cls)
        objeto._preparar(decodificador, generador)
        return objeto

    def _preparar(self, decodificador, generador):
        self.generador = generador
        self.torch = generador.torch
        self.gpt = generador.gpt
        self.dispositivo = generador.dispositivo
        self.decodificador = decodificador.to(self.dispositivo).eval().requires_grad_(False)
        self.sr = decodificador.output_sample_rate
        self.sr_hablante = decodificador.speaker_encoder_audio_config["sample_rate"]

    def embedding_hablante(self, muestras) -> np.ndarray:
        """Voz de referencia a 22 050 Hz -> vector de hablante [512], como `Xtts.get_speaker_embedding`."""
        import torchaudio

        torch = self.torch
        audio = torch.from_numpy(np.ascontiguousarray(muestras, dtype=np.float32))[None]
        audio = audio[:, : config.SR * config.LONGITUD_MAXIMA_CONDICIONAMIENTO_S]
        with torch.inference_mode():
            audio_16k = torchaudio.functional.resample(audio, config.SR, self.sr_hablante)
            vector = self.decodificador.speaker_encoder.forward(audio_16k.to(self.dispositivo), l2_norm=True)
        return vector[0].float().cpu().numpy()

    def latentes(self, vectores, ids, tokens) -> np.ndarray:
        """Segunda pasada del GPT-2 sobre los tokens ya generados -> latentes [N, 1024].

        Es la llamada de `Xtts.inference`: `gpt(texto, largo, códigos, largo_esperado,
        cond_latents=..., return_latent=True)`.
        """
        torch = self.torch
        codigos = np.asarray(tokens, dtype=np.int64)
        if codigos.ndim != 1 or codigos.size == 0:
            raise ValueError("No hay tokens de audio que sintetizar.")
        if not self.generador.cerrojo.acquire(blocking=False):
            raise GeneradorOcupado("El GPT-2 está ocupado. Espera a que termine.")
        try:
            cond = torch.from_numpy(np.asarray(vectores, dtype=np.float32))[None].to(self.dispositivo)
            texto = torch.IntTensor([int(i) for i in ids])[None].to(self.dispositivo)
            codigos = torch.from_numpy(codigos)[None].to(self.dispositivo)
            largo_esperado = torch.tensor([codigos.shape[-1] * self.gpt.code_stride_len], device=self.dispositivo)
            largo_texto = torch.tensor([texto.shape[-1]], device=self.dispositivo)
            with torch.inference_mode():
                latentes = self.gpt(texto, largo_texto, codigos, largo_esperado, cond_latents=cond,
                                    return_attentions=False, return_latent=True)
        finally:
            self.generador.cerrojo.release()
        return latentes[0].float().cpu().numpy()

    def onda(self, latentes, embedding) -> np.ndarray:
        """Latentes [N, 1024] + vector de hablante [512] -> muestras a 24 000 Hz."""
        torch = self.torch
        z = torch.from_numpy(np.asarray(latentes, dtype=np.float32))[None].to(self.dispositivo)
        g = torch.from_numpy(np.asarray(embedding, dtype=np.float32)).reshape(1, -1, 1).to(self.dispositivo)
        with torch.inference_mode():
            salida = self.decodificador(z, g=g)
        return salida.float().cpu().squeeze().numpy()

    def mel_de_salida(self, onda) -> np.ndarray:
        """El mel de la voz generada, con los parámetros de la etapa 1 para poder compararlos."""
        en_22k = librosa.resample(np.asarray(onda, dtype=np.float32), orig_sr=self.sr, target_sr=config.SR)
        return caracteristicas.mel_espectrograma(en_22k)

    def sintetizar(self, vectores, ids, tokens, embedding, fases=None) -> Sintesis:
        """Ejecuta la etapa 4 completa.

        1. Latentes: segunda pasada del GPT-2 sobre los tokens.
        2. HiFi-GAN: latentes + vector de hablante -> onda a 24 kHz.
        3. Mel de esa onda y, si hay DVAE, el mel que describen los tokens.
        """
        tiempos = {}
        reloj = time.perf_counter()
        latentes = self.latentes(vectores, ids, tokens)
        tiempos["latentes"] = (time.perf_counter() - reloj) * 1000

        reloj = time.perf_counter()
        onda = self.onda(latentes, embedding)
        tiempos["hifigan"] = (time.perf_counter() - reloj) * 1000

        reloj = time.perf_counter()
        mel_salida = self.mel_de_salida(onda)
        mel_dvae = None
        if fases is not None:
            try:
                mel_dvae = fases.mel_desde_tokens(tokens)
            except ValueError:  # solo tokens especiales: no hay nada que decodificar
                mel_dvae = None
        tiempos["mels"] = (time.perf_counter() - reloj) * 1000
        return Sintesis(onda=onda, sr=self.sr, latentes=latentes,
                        embedding=np.asarray(embedding, dtype=np.float32),
                        mel_salida=mel_salida, mel_dvae=mel_dvae, tiempos_ms=tiempos)
