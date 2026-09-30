"""Fases audibles: cómo suenan el mel, los tokens del DVAE y los del GPT-2.

Es audio crudo a propósito. Griffin-Lim estima la fase que el mel descartó, así
que suena metálico, y el mel no guarda nada por encima de 8 kHz, así que suena
apagado. La calidad final la da el HiFi-GAN de la etapa 4.
"""

from __future__ import annotations

import io

import librosa
import numpy as np
import soundfile as sf

from . import config
from .identidad import ModeloNoDisponible, comprobar_archivos, importar_torch

ITERACIONES_GRIFFIN_LIM = 32
N_FFT_DVAE = 1024
"""El DVAE se entrenó con `TorchMelSpectrogram` por omisión: n_fft 1024, no 2048."""
CODIGOS_DVAE = 1024
MUESTRAS_POR_CODIGO = 1024
"""Cada código del DVAE son 4 frames de mel de 256 muestras: el `code_stride_len`."""

FASES = ("mel", "tokens_referencia", "tokens_texto")


def mel_a_audio(log_mel, n_fft, iteraciones=ITERACIONES_GRIFFIN_LIM) -> np.ndarray:
    """Log-mel [80, T] sin normalizar -> muestras a 22 050 Hz, con Griffin-Lim."""
    potencia = np.exp(np.asarray(log_mel, dtype=np.float32))
    return librosa.feature.inverse.mel_to_audio(
        potencia, sr=config.SR, n_fft=n_fft, hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH, power=config.MEL_POWER, n_iter=iteraciones,
        htk=config.MEL_ESCALA == "htk", norm=config.MEL_NORM,
        fmin=config.MEL_FMIN, fmax=config.MEL_FMAX)


def audio_desde_mel_referencia(mel) -> np.ndarray:
    """Fase 2: el mel de la etapa 1 (n_fft 2048) convertido de nuevo en sonido."""
    return mel_a_audio(mel, config.N_FFT)


def a_wav(muestras, pico=0.9) -> bytes:
    """Muestras -> bytes de un WAV PCM de 16 bits, con el pico normalizado."""
    muestras = np.nan_to_num(np.asarray(muestras, dtype=np.float32))
    maximo = float(np.max(np.abs(muestras))) if muestras.size else 0.0
    if maximo > 0:
        muestras = muestras * (pico / maximo)
    salida = io.BytesIO()
    sf.write(salida, muestras, config.SR, format="WAV", subtype="PCM_16")
    return salida.getvalue()


class ReconstructorFases:
    """DVAE oficial de XTTS-v2 para pasar voz a tokens y tokens a mel."""

    ESENCIALES = ("encoder.", "decoder.", "codebook.embed")
    """Lo demás del estado son acumuladores de entrenamiento que el decodificado no usa."""

    def __init__(self, carpeta, normas, dispositivo="cpu"):
        carpeta = comprobar_archivos(carpeta, ("dvae.pth",))
        torch = importar_torch()
        from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
        from TTS.tts.layers.xtts.dvae import DiscreteVAE

        # Mismos argumentos que el entrenador oficial (gpt_trainer.py).
        dvae = DiscreteVAE(channels=80, normalization=None, positional_dims=1,
                           num_tokens=CODIGOS_DVAE, codebook_dim=512, hidden_dim=512,
                           num_resnet_blocks=3, kernel_size=3, num_layers=2,
                           use_transposed_convs=False)
        try:
            estado = torch.load(carpeta / "dvae.pth", map_location="cpu", weights_only=True)
            # strict=False como el entrenador oficial, pero sin aceptar huecos que importen.
            faltan = dvae.load_state_dict(estado, strict=False).missing_keys
        except (RuntimeError, OSError, EOFError, ValueError) as exc:
            raise ModeloNoDisponible("No se pudo leer dvae.pth. Detalle: " + str(exc)) from exc
        imprescindibles = [k for k in faltan if k.startswith(self.ESENCIALES)]
        if imprescindibles:
            raise ModeloNoDisponible("dvae.pth no trae pesos imprescindibles: "
                                     + ", ".join(imprescindibles[:3]))

        normas = np.asarray(normas, dtype=np.float32)
        self.torch = torch
        self.dispositivo = dispositivo
        self.normas = normas
        self.dvae = dvae.to(dispositivo).eval().requires_grad_(False)
        # El mel con el que se entrenó el DVAE: log con piso y dividido por mel_stats.
        self.mel_dvae = TorchMelSpectrogram(mel_norm_file=None, sampling_rate=config.SR)
        self.mel_dvae.mel_norms = torch.from_numpy(normas)

    def tokens_de_referencia(self, muestras) -> np.ndarray:
        """Fase 3, ida: voz [n] -> códigos del DVAE [n / 1024]."""
        torch = self.torch
        senal = torch.from_numpy(np.ascontiguousarray(muestras, dtype=np.float32))[None]
        with torch.inference_mode():
            mel = self.mel_dvae(senal.to(self.dispositivo))
            codigos = self.dvae.get_codebook_indices(mel)
        return codigos[0].cpu().numpy().astype(np.int64)

    def mel_desde_tokens(self, tokens) -> np.ndarray:
        """Códigos -> log-mel [80, 4 * N] sin normalizar.

        Los tokens de inicio (1024) y fin (1025) de audio no son códigos del DVAE y
        se descartan.
        """
        torch = self.torch
        codigos = np.asarray(tokens, dtype=np.int64)
        codigos = codigos[(codigos >= 0) & (codigos < CODIGOS_DVAE)]
        if codigos.size == 0:
            raise ValueError("No hay tokens de audio que decodificar.")
        with torch.inference_mode():
            mel, _ = self.dvae.decode(torch.from_numpy(codigos)[None].to(self.dispositivo))
        # El DVAE devuelve el mel dividido por mel_stats; se deshace esa división.
        return mel[0].float().cpu().numpy() * self.normas[:, None]

    def audio_desde_tokens(self, tokens) -> np.ndarray:
        """Fases 3 (vuelta) y 4: códigos -> mel -> sonido con Griffin-Lim."""
        return mel_a_audio(self.mel_desde_tokens(tokens), N_FFT_DVAE)
