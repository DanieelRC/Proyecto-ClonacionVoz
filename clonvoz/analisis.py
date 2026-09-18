"""
Orquestación de la etapa 1: de un .wav a las cuatro características.

Este módulo es lo que comparten la línea de comandos y el servidor web, de modo
que el resultado sea idéntico por los dos caminos y la lógica exista una sola vez.
Aquí también se cronometra cada paso (RF-20) y se genera el identificador de
corrida (RF-08), que en la etapa 5 servirá para correlacionar los eventos de
activación con su audio.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from . import caracteristicas, config
from .audio import AudioCargado, Validacion, cargar_audio, validar_audio


class AudioRechazado(Exception):
    """
    El audio se leyó bien, pero no cumple las reglas de validación (RF-03).

    Lleva la validación completa para poder mostrar todos los motivos a la vez en
    lugar de solo el primero.
    """

    def __init__(self, validacion: Validacion):
        self.validacion = validacion
        super().__init__(" ".join(validacion.errores))


@dataclass
class Analisis:
    """Todo lo que la etapa 1 sabe de un audio, listo para graficar o serializar."""

    id_corrida: str
    audio: AudioCargado
    validacion: Validacion
    metodo_f0: str

    tiempo: np.ndarray
    amplitud: np.ndarray

    espectrograma_db: np.ndarray
    frecuencias: np.ndarray

    mel: np.ndarray

    f0: np.ndarray
    voz: np.ndarray
    confianza: np.ndarray

    tiempo_frames: np.ndarray
    tiempos_ms: dict[str, float] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.mel.shape[1])

    @property
    def formas(self) -> dict[str, list[int]]:
        """
        Forma de cada tensor intermedio.

        RF-11 lo pedirá formalmente en la etapa 5, junto al diagrama; empezar a
        reportarlas desde ahora es gratis y es lo que hace que la visualización
        explique algo en vez de solo adornar.
        """
        return {
            "forma_de_onda": [int(self.amplitud.shape[0])],
            "espectrograma": [int(d) for d in self.espectrograma_db.shape],
            "mel": [int(d) for d in self.mel.shape],
            "f0": [int(self.f0.shape[0])],
        }

    @property
    def metadatos(self) -> dict:
        return {
            "duracion_s": round(self.audio.duracion_s, 3),
            "sr_trabajo": self.audio.sr,
            "sr_original": self.audio.sr_original,
            "canales_original": self.audio.canales_original,
            "formato": self.audio.formato,
            "subtipo": self.audio.subtipo,
            "n_muestras": self.audio.n_muestras,
            "n_frames": self.n_frames,
            "metodo_f0": self.metodo_f0,
        }

    @property
    def estadisticas(self) -> dict:
        """Resumen numérico de las cuatro representaciones."""
        f0_con_voz = self.f0[np.isfinite(self.f0)]
        tiene_voz = f0_con_voz.size > 0
        return {
            "rms": round(self.audio.rms, 5),
            "pico": round(self.audio.pico, 4),
            "mel_min": round(float(self.mel.min()), 3),
            "mel_max": round(float(self.mel.max()), 3),
            "mel_media": round(float(self.mel.mean()), 3),
            "f0_media_hz": round(float(np.mean(f0_con_voz)), 1) if tiene_voz else None,
            "f0_min_hz": round(float(np.min(f0_con_voz)), 1) if tiene_voz else None,
            "f0_max_hz": round(float(np.max(f0_con_voz)), 1) if tiene_voz else None,
            "porcentaje_con_voz": round(100.0 * float(np.mean(self.voz)), 1),
        }

    def como_dict(self) -> dict:
        """Resumen serializable a JSON. No incluye tensores ni imágenes."""
        return {
            "id_corrida": self.id_corrida,
            "metadatos": self.metadatos,
            "formas": self.formas,
            "estadisticas": self.estadisticas,
            "tiempos_ms": {k: round(v, 1) for k, v in self.tiempos_ms.items()},
            "advertencias": list(self.validacion.advertencias),
            "parametros_mel": {
                "sr": config.SR,
                "n_fft": config.N_FFT,
                "hop_length": config.HOP_LENGTH,
                "win_length": config.WIN_LENGTH,
                "n_mels": config.N_MELS,
                "fmin": config.MEL_FMIN,
                "fmax": config.MEL_FMAX,
                "escala": config.MEL_ESCALA,
                "norm": config.MEL_NORM,
            },
        }


@contextmanager
def _cronometro(tiempos: dict[str, float], clave: str):
    """Guarda en `tiempos` los milisegundos que tardó el bloque."""
    inicio = time.perf_counter()
    try:
        yield
    finally:
        tiempos[clave] = (time.perf_counter() - inicio) * 1000.0


def analizar(
    fuente,
    metodo_f0: str = config.METODO_F0_POR_OMISION,
    id_corrida: str | None = None,
) -> Analisis:
    """
    Ejecuta la etapa 1 completa sobre un audio.

    `fuente` es lo que acepta `cargar_audio`: una ruta, un archivo abierto o los
    bytes del .wav.

    Levanta `AudioInvalido` si el archivo no se puede leer y `AudioRechazado` si
    se lee pero no pasa la validación (RF-03).
    """
    tiempos: dict[str, float] = {}

    with _cronometro(tiempos, "carga"):
        audio = cargar_audio(fuente)

    validacion = validar_audio(audio)
    if not validacion.valido:
        raise AudioRechazado(validacion)

    with _cronometro(tiempos, "forma_de_onda"):
        tiempo, amplitud = caracteristicas.forma_de_onda(audio.muestras)

    with _cronometro(tiempos, "espectrograma"):
        espectrograma_db, frecuencias = caracteristicas.espectrograma_stft(audio.muestras)

    with _cronometro(tiempos, "mel"):
        mel = caracteristicas.mel_espectrograma(audio.muestras)

    with _cronometro(tiempos, "f0"):
        f0, voz, confianza = caracteristicas.contorno_f0(audio.muestras, metodo=metodo_f0)

    return Analisis(
        id_corrida=id_corrida or str(uuid.uuid4()),
        audio=audio,
        validacion=validacion,
        metodo_f0=metodo_f0,
        tiempo=tiempo,
        amplitud=amplitud,
        espectrograma_db=espectrograma_db,
        frecuencias=frecuencias,
        mel=mel,
        f0=f0,
        voz=voz,
        confianza=confianza,
        tiempo_frames=caracteristicas.eje_de_tiempo_frames(int(mel.shape[1])),
        tiempos_ms=tiempos,
    )
