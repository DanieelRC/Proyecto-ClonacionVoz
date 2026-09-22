"""Carga y validación del audio de referencia en mono float32 a 22 050 Hz."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import librosa
import numpy as np
import soundfile as sf

from . import config


class AudioInvalido(Exception):
    """El audio no se pudo leer o no tiene un formato válido."""


@dataclass
class AudioCargado:
    """Audio de referencia normalizado para análisis."""

    muestras: np.ndarray
    sr: int
    sr_original: int
    canales_original: int
    formato: str
    subtipo: str

    @property
    def n_muestras(self) -> int:
        return int(self.muestras.shape[0])

    @property
    def duracion_s(self) -> float:
        return self.n_muestras / self.sr

    @property
    def rms(self) -> float:
        """Nivel medio RMS de la señal."""
        if self.n_muestras == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.muestras, dtype=np.float64))))

    @property
    def pico(self) -> float:
        """Amplitud máxima absoluta."""
        if self.n_muestras == 0:
            return 0.0
        return float(np.max(np.abs(self.muestras)))


@dataclass
class Validacion:
    """Resultado de la validación del audio con errores y advertencias."""

    errores: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)

    @property
    def valido(self) -> bool:
        return not self.errores

    def como_dict(self) -> dict:
        return {
            "valido": self.valido,
            "errores": list(self.errores),
            "advertencias": list(self.advertencias),
        }


def cargar_audio(fuente) -> AudioCargado:
    """Lee un archivo WAV y lo convierte a mono float32 a 22 050 Hz."""
    if isinstance(fuente, (bytes, bytearray, memoryview)):
        fuente = io.BytesIO(bytes(fuente))
    elif hasattr(fuente, "seek"):
        fuente.seek(0)

    try:
        with sf.SoundFile(fuente) as archivo:
            formato = archivo.format
            subtipo = archivo.subtype
            sr_original = int(archivo.samplerate)
            canales = int(archivo.channels)
            datos = archivo.read(dtype="float32", always_2d=True)
    except AudioInvalido:
        raise
    except Exception as exc:
        raise AudioInvalido(
            "No se pudo leer el archivo como audio. Asegúrate de que sea un .wav "
            "válido y que no esté incompleto o dañado."
        ) from exc

    if formato.upper() not in config.FORMATOS_ACEPTADOS:
        raise AudioInvalido(
            f"El archivo está en formato {formato}, no WAV. Convierte el audio a "
            ".wav antes de subirlo (cambiarle la extensión al nombre no lo convierte)."
        )

    if datos.size == 0:
        raise AudioInvalido("El archivo de audio está vacío: no contiene ninguna muestra.")

    # Mezclar canales a mono
    muestras = datos.mean(axis=1) if canales > 1 else datos[:, 0]

    # Remuestrear a la frecuencia objetivo si difiere
    if sr_original != config.SR:
        muestras = librosa.resample(
            np.ascontiguousarray(muestras, dtype=np.float32),
            orig_sr=sr_original,
            target_sr=config.SR,
        )

    return AudioCargado(
        muestras=np.ascontiguousarray(muestras, dtype=np.float32),
        sr=config.SR,
        sr_original=sr_original,
        canales_original=canales,
        formato=formato,
        subtipo=subtipo,
    )


def validar_audio(audio: AudioCargado) -> Validacion:
    """Valida duración, nivel RMS y parámetros del audio cargado."""
    validacion = Validacion()
    duracion = audio.duracion_s

    if duracion < config.DURACION_MINIMA_S:
        validacion.errores.append(
            f"El audio dura {duracion:.1f} s y se necesitan al menos "
            f"{config.DURACION_MINIMA_S:.0f} s. Graba o sube una muestra más larga."
        )
    elif duracion > config.DURACION_MAXIMA_S:
        validacion.errores.append(
            f"El audio dura {duracion:.1f} s y el máximo son "
            f"{config.DURACION_MAXIMA_S:.0f} s. Recorta la muestra antes de subirla."
        )
    elif duracion < config.DURACION_RECOMENDADA_MINIMA_S:
        validacion.advertencias.append(
            f"El audio dura {duracion:.1f} s. Se puede analizar, pero la clonación "
            f"funciona mejor con referencias de "
            f"{config.DURACION_RECOMENDADA_MINIMA_S:.0f} a "
            f"{config.DURACION_RECOMENDADA_MAXIMA_S:.0f} s."
        )

    if audio.rms < config.RMS_MINIMO:
        validacion.errores.append(
            "El audio está en silencio o casi en silencio. Revisa que el micrófono "
            "correcto esté seleccionado y que no esté silenciado."
        )
    elif audio.pico > 0.99:
        validacion.advertencias.append(
            "El audio llega a saturarse (la señal toca el máximo). Conviene grabar "
            "más lejos del micrófono o bajar la ganancia de entrada."
        )

    if audio.sr_original != config.SR:
        validacion.advertencias.append(
            f"El audio venía a {audio.sr_original} Hz y se remuestreó a "
            f"{config.SR} Hz, que es la frecuencia que usa XTTS-v2. Grabar "
            f"directamente a {config.SR} Hz evita esta conversión."
        )

    return validacion


def nombre_seguro(nombre: str | None) -> str:
    """Devuelve el nombre base seguro del archivo para evitar rutas externas."""
    if not nombre:
        return "audio.wav"
    return os.path.basename(str(nombre).replace("\\", "/")) or "audio.wav"
