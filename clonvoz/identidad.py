"""Codificador de identidad de hablante (Perceiver Resampler de XTTS-v2)."""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ModeloNoDisponible(RuntimeError):
    """El modelo no está disponible o falta su configuración/pesos."""


@dataclass
class Identidad:
    """Paquete de vectores de identidad y mapas de atención calculados."""

    id_corrida: str
    vectores: np.ndarray
    atencion: np.ndarray
    atencion_consultas: np.ndarray
    fragmentos: list[dict]
    tiempos_ms: dict
    advertencias: list[str]
    dispositivo: str
    capas_atencion: list | None = None

    def como_dict(self):
        """Genera un resumen serializable a JSON con estadísticas y distribución de atención."""
        capas = self.capas_atencion or [np.concatenate(
            (self.atencion_consultas[0], self.atencion), axis=1)]
        resumen_atencion = []
        for indice, capa in enumerate(capas):
            resumen_atencion.append({
                "capa": ("Primera capa" if indice == 0 else "Última capa")
                        if self.capas_atencion else "Última capa",
                "audio_porcentaje": float(capa[:, 32:].sum(axis=1, dtype=np.float64).mean() * 100),
                "consultas_porcentaje": float(capa[:, :32].sum(axis=1, dtype=np.float64).mean() * 100),
            })
        return {
            "resumen_atencion": resumen_atencion,
            "id_corrida": self.id_corrida,
            "formas": {"identidad": list(self.vectores.shape),
                       "atencion": list(self.atencion.shape)},
            "fragmentos": self.fragmentos,
            "tiempos_ms": {k: round(v, 2) for k, v in self.tiempos_ms.items()},
            "advertencias": self.advertencias,
            "dispositivo": self.dispositivo,
            "captura": "Última capa del Perceiver, promedio de sus 8 cabezas; sin renormalizar.",
            "estadisticas": {"minimo": float(self.vectores.min()),
                             "maximo": float(self.vectores.max()),
                             "media": float(self.vectores.mean()),
                             "desviacion": float(self.vectores.std())},
        }


def carpeta_modelo() -> Path:
    """Ruta a la carpeta del modelo (por defecto modelos/xtts_v2 o CLONVOZ_MODELO)."""
    return Path(os.environ.get("CLONVOZ_MODELO", "modelos/xtts_v2"))


class CodificadorIdentidad:
    """Carga los módulos preentrenados de XTTS-v2 y gestiona la inferencia del Perceiver."""

    def __init__(self, carpeta, dispositivo="auto"):
        carpeta = Path(carpeta)
        if not all((carpeta / nombre).is_file() for nombre in ("config.json", "model.pth")):
            raise ModeloNoDisponible(
                "Faltan config.json y/o model.pth de XTTS-v2. "
                "Consulta la preparación de la etapa 2 en el README."
            )

        try:
            import torch
            from TTS.tts.layers.tortoise.autoregressive import ConditioningEncoder
            from TTS.tts.layers.xtts.perceiver_encoder import PerceiverResampler
            from TTS.config.shared_configs import BaseDatasetConfig
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
        except (ImportError, OSError) as exc:
            raise ModeloNoDisponible(
                "No se pudieron cargar PyTorch y los módulos de coqui-tts. "
                "Usa el entorno de la etapa 2 indicado en el README."
            ) from exc

        self.torch = torch
        if dispositivo not in ("auto", "cpu", "cuda"):
            raise ModeloNoDisponible("El dispositivo debe ser auto, cpu o cuda.")
        self.dispositivo = ("cuda" if torch.cuda.is_available() else "cpu") if dispositivo == "auto" else dispositivo
        if self.dispositivo == "cuda" and not torch.cuda.is_available():
            raise ModeloNoDisponible("CUDA no está disponible. Selecciona cpu o auto.")

        try:
            datos = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
            args = datos["model_args"]
            if (args["gpt_n_model_channels"] != 1024 or args["gpt_n_heads"] != 16
                    or not args["gpt_use_perceiver_resampler"]):
                raise ValueError("La configuración no corresponde al Perceiver de XTTS-v2.")

            with torch.serialization.safe_globals([XttsConfig, XttsArgs, XttsAudioConfig, BaseDatasetConfig]):
                checkpoint = torch.load(carpeta / "model.pth", map_location="cpu", weights_only=True, mmap=True)
            pesos = checkpoint["model"]
            pesos = {k.removeprefix("xtts."): v for k, v in pesos.items()}

            self.encoder = ConditioningEncoder(80, 1024, num_attn_heads=16)
            self.perceiver = PerceiverResampler(dim=1024, depth=2, dim_context=1024,
                                               num_latents=32, dim_head=64, heads=8,
                                               ff_mult=4, use_flash_attn=False)

            for modulo, prefijo in ((self.encoder, "gpt.conditioning_encoder."),
                                    (self.perceiver, "gpt.conditioning_perceiver.")):
                seleccion = {k[len(prefijo):]: v for k, v in pesos.items() if k.startswith(prefijo)}
                modulo.load_state_dict(seleccion, strict=True)
                modulo.to(self.dispositivo).eval().requires_grad_(False)

            self.normas = pesos["mel_stats"].float().cpu().numpy().copy()
            if self.normas.shape != (80,) or not np.isfinite(self.normas).all() or (self.normas == 0).any():
                raise ValueError("Las estadísticas Mel del checkpoint no son válidas.")
        except (KeyError, ValueError, RuntimeError, OSError, EOFError, pickle.UnpicklingError) as exc:
            raise ModeloNoDisponible(
                "No se pudo cargar el checkpoint compatible de XTTS-v2. "
                "Comprueba los archivos oficiales y sus versiones. Detalle: " + str(exc)
            ) from exc

        self.cerrojo = threading.Lock()

    def procesar_mel(self, mel, *, diagnostico=None):
        """Procesa el mel [80, T] y extrae los vectores de latentes y matrices de atención."""
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] == 0 or not np.isfinite(mel).all():
            raise ValueError("Se esperaba un mel finito de forma [80, T], con T mayor que cero.")

        torch = self.torch
        capturas = []

        def capturar(_modulo, _entrada, salida):
            capturas.append(salida.detach().mean(dim=1)[0].cpu().numpy().copy())

        with self.cerrojo, torch.inference_mode():
            hooks = [capa[0].attend.attn_dropout.register_forward_hook(capturar)
                     for capa in self.perceiver.layers]
            try:
                entrada = torch.from_numpy(mel / self.normas[:, None]).unsqueeze(0).to(self.dispositivo)
                contexto = self.encoder(entrada).transpose(1, 2)
                vectores = self.perceiver(contexto)[0].cpu().numpy().copy()
            finally:
                for hook in hooks:
                    hook.remove()

        if len(capturas) != 2 or any(c.shape != (32, 32 + mel.shape[1]) for c in capturas):
            raise ModeloNoDisponible("La atención del módulo instalado tiene una forma inesperada.")
        if vectores.shape != (32, 1024) or not np.isfinite(vectores).all():
            raise ModeloNoDisponible("El modelo produjo vectores inválidos.")

        if diagnostico is not None:
            diagnostico.extend(capturas)

        return vectores, capturas[-1][:, 32:], capturas[-1][:, :32]


def identidad_desde_mel(mel, modelo, id_corrida, duracion_s, advertencias=()):
    """Ejecuta el codificador sobre el mel calculado y empaqueta el resultado en Identidad."""
    inicio = time.perf_counter()
    capas = []
    if isinstance(modelo, CodificadorIdentidad):
        vectores, atencion, propias = modelo.procesar_mel(mel, diagnostico=capas)
    else:
        vectores, atencion, propias = modelo.procesar_mel(mel)
    transcurrido = (time.perf_counter() - inicio) * 1000

    intervalos = [{"inicio_s": 0.0, "fin_s": duracion_s,
                   "columna_inicio": 0, "frames": int(mel.shape[1]),
                   "masa_atencion_audio": float(atencion.sum(axis=1).mean())}]
    return Identidad(id_corrida, vectores, atencion, propias[None, :, :], intervalos,
                     {"codificacion": transcurrido, "total": transcurrido},
                     list(advertencias), modelo.dispositivo, capas or None)


def analizar_identidad(fuente, modelo):
    """Encadena el análisis de etapa 1 y la extracción de identidad para CLI o scripts."""
    from .analisis import analizar

    etapa1 = analizar(fuente)
    return identidad_desde_mel(etapa1.mel, modelo, etapa1.id_corrida,
                               etapa1.audio.duracion_s, etapa1.validacion.advertencias)
