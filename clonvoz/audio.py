"""
Carga y validación del audio de referencia (RF-02, RF-03).

Este módulo es la única puerta de entrada del audio al resto del sistema: sea que
venga de un archivo en disco (CLI) o de una carga del navegador (servidor), sale
de aquí siempre igual —mono, float32 y a 22 050 Hz— para que ningún módulo de
etapas posteriores tenga que preguntarse en qué formato le llegó.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import librosa
import numpy as np
import soundfile as sf

from . import config


class AudioInvalido(Exception):
    """
    El audio no se pudo leer, o no es un WAV.

    Su mensaje está escrito para mostrarse tal cual al usuario, en español y sin
    jerga: es el mensaje claro que pide RF-03.
    """


@dataclass
class AudioCargado:
    """Audio de referencia ya normalizado a las condiciones de análisis."""

    muestras: np.ndarray
    """Señal mono en float32, remuestreada a `config.SR`."""

    sr: int
    """Frecuencia de muestreo de trabajo (siempre `config.SR`)."""

    sr_original: int
    """Frecuencia de muestreo con la que venía el archivo."""

    canales_original: int
    """Canales con los que venía el archivo (se promedian a mono)."""

    formato: str
    """Contenedor reportado por libsndfile, por ejemplo "WAV"."""

    subtipo: str
    """Codificación de las muestras, por ejemplo "PCM_16"."""

    @property
    def n_muestras(self) -> int:
        return int(self.muestras.shape[0])

    @property
    def duracion_s(self) -> float:
        # Cuántas muestras hay, entre cuántas caben en un segundo.
        return self.n_muestras / self.sr

    @property
    def rms(self) -> float:
        """Nivel medio de la señal. Sirve para detectar audio en silencio."""
        if self.n_muestras == 0:
            return 0.0
        # Raíz de la media de los cuadrados: eleva al cuadrado para que los valores
        # negativos no cancelen a los positivos, promedia y deshace el cuadrado. Da
        # el "volumen típico" del audio, no su pico. El acumulado va en float64
        # porque sumar 176 400 cuadrados en float32 pierde precisión.
        return float(np.sqrt(np.mean(np.square(self.muestras, dtype=np.float64))))

    @property
    def pico(self) -> float:
        """Amplitud máxima absoluta. Cerca de 1.0 sugiere saturación."""
        if self.n_muestras == 0:
            return 0.0
        # El valor más extremo, sin importar si fue hacia arriba o hacia abajo.
        return float(np.max(np.abs(self.muestras)))


@dataclass
class Validacion:
    """Resultado de validar el audio: qué lo descalifica y qué solo lo empeora."""

    errores: list[str] = field(default_factory=list)
    """Motivos por los que el audio no se puede procesar. Si hay uno, se rechaza."""

    advertencias: list[str] = field(default_factory=list)
    """Avisos que no impiden continuar, pero que degradan el resultado."""

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
    """
    Lee un WAV y lo deja listo para el análisis.

    `fuente` puede ser una ruta, un objeto de archivo abierto o los bytes crudos
    del archivo (el caso de la carga por HTTP, que nunca toca el disco).

    Levanta `AudioInvalido` si no se puede leer o si no es un WAV.

    La receta, paso a paso:
      1. Aceptar las tres formas de entrada y dejarlas como algo que se pueda leer.
      2. Abrir el archivo y sacar sus muestras y sus datos técnicos.
      3. Rechazarlo si el contenedor no es WAV de verdad.
      4. Mezclar los canales a uno solo.
      5. Remuestrear a 22 050 Hz si venía a otra frecuencia.
    """
    # 1. Los bytes crudos se envuelven en un BytesIO, que se comporta como un
    #    archivo sin existir en disco. Si ya es un archivo abierto se rebobina al
    #    inicio, porque puede venir a medio leer y entonces faltarían muestras.
    if isinstance(fuente, (bytes, bytearray, memoryview)):
        fuente = io.BytesIO(bytes(fuente))
    elif hasattr(fuente, "seek"):
        fuente.seek(0)

    try:
        with sf.SoundFile(fuente) as archivo:
            # 2. La cabecera del archivo declara todo esto antes de leer una sola
            #    muestra; se guarda para reportarlo y para decidir si hay que
            #    remuestrear o mezclar canales.
            formato = archivo.format        # el contenedor: "WAV", "MP3", "FLAC"…
            subtipo = archivo.subtype       # cómo están codificadas las muestras
            sr_original = int(archivo.samplerate)
            canales = int(archivo.channels)
            # always_2d fuerza la forma [n, canales] incluso en audio mono, para no
            # tener que escribir dos caminos distintos más abajo.
            datos = archivo.read(dtype="float32", always_2d=True)
    except AudioInvalido:
        raise
    except Exception as exc:  # libsndfile levanta tipos distintos según el fallo
        raise AudioInvalido(
            "No se pudo leer el archivo como audio. Asegúrate de que sea un .wav "
            "válido y que no esté incompleto o dañado."
        ) from exc

    # 3. El contenedor real, no la extensión: libsndfile también lee MP3, así que un
    #    archivo renombrado a .wav se abriría sin error (RF-02).
    if formato.upper() not in config.FORMATOS_ACEPTADOS:
        raise AudioInvalido(
            f"El archivo está en formato {formato}, no WAV. Convierte el audio a "
            ".wav antes de subirlo (cambiarle la extensión al nombre no lo convierte)."
        )

    if datos.size == 0:
        raise AudioInvalido("El archivo de audio está vacío: no contiene ninguna muestra.")

    # 4. [n, canales] -> [n]. Promediar los canales es la mezcla a mono: en estéreo
    #    ambos llevan casi la misma voz, y XTTS-v2 trabaja con una sola pista. Con un
    #    solo canal basta con extraer la columna, sin promediar nada.
    muestras = datos.mean(axis=1) if canales > 1 else datos[:, 0]

    # 5. [n] -> [n * 22050/sr_original]. Un audio de 44 100 Hz tiene el doble de
    #    muestras por segundo de las que espera XTTS-v2; remuestrear recalcula la
    #    señal a 22 050 Hz conservando el sonido, no tirando muestras a la mitad.
    if sr_original != config.SR:
        muestras = librosa.resample(
            np.ascontiguousarray(muestras, dtype=np.float32),
            orig_sr=sr_original,
            target_sr=config.SR,
        )

    return AudioCargado(
        # ascontiguousarray deja las muestras seguidas en memoria: torch lo exige
        # para envolverlas sin copiarlas cuando se calcule el mel.
        muestras=np.ascontiguousarray(muestras, dtype=np.float32),
        sr=config.SR,
        sr_original=sr_original,
        canales_original=canales,
        formato=formato,
        subtipo=subtipo,
    )


def validar_audio(audio: AudioCargado) -> Validacion:
    """
    Aplica las reglas de RF-03 sobre un audio ya cargado.

    Separa lo que impide procesar (duración fuera de rango, silencio total) de lo
    que solo empeora el resultado (referencia corta, señal saturada), para no
    bloquear al usuario por algo que sí puede seguir usando.
    """
    validacion = Validacion()
    duracion = audio.duracion_s

    # Las tres ramas son excluyentes a propósito: si el audio ya es demasiado corto
    # no tiene sentido además advertirle que es más corto de lo recomendable.
    if duracion < config.DURACION_MINIMA_S:
        # Menos de 3 s no alcanzan para caracterizar una voz.
        validacion.errores.append(
            f"El audio dura {duracion:.1f} s y se necesitan al menos "
            f"{config.DURACION_MINIMA_S:.0f} s. Graba o sube una muestra más larga."
        )
    elif duracion > config.DURACION_MAXIMA_S:
        # Más de 30 s los recorta el propio XTTS-v2, así que solo alargarían el
        # análisis sin mejorar el resultado.
        validacion.errores.append(
            f"El audio dura {duracion:.1f} s y el máximo son "
            f"{config.DURACION_MAXIMA_S:.0f} s. Recorta la muestra antes de subirla."
        )
    elif duracion < config.DURACION_RECOMENDADA_MINIMA_S:
        # Entre 3 y 6 s se puede analizar, pero la clonación sale peor: advertencia.
        validacion.advertencias.append(
            f"El audio dura {duracion:.1f} s. Se puede analizar, pero la clonación "
            f"funciona mejor con referencias de "
            f"{config.DURACION_RECOMENDADA_MINIMA_S:.0f} a "
            f"{config.DURACION_RECOMENDADA_MAXIMA_S:.0f} s."
        )

    if audio.rms < config.RMS_MINIMO:
        # Nivel medio casi nulo: micrófono equivocado, silenciado o desconectado.
        validacion.errores.append(
            "El audio está en silencio o casi en silencio. Revisa que el micrófono "
            "correcto esté seleccionado y que no esté silenciado."
        )
    elif audio.pico > 0.99:
        # El pico toca el techo del formato: la onda viene recortada arriba y abajo,
        # y esa distorsión se copiaría a la voz clonada.
        validacion.advertencias.append(
            "El audio llega a saturarse (la señal toca el máximo). Conviene grabar "
            "más lejos del micrófono o bajar la ganancia de entrada."
        )

    if audio.sr_original != config.SR:
        # No es un problema, pero conviene que se sepa que hubo una conversión.
        validacion.advertencias.append(
            f"El audio venía a {audio.sr_original} Hz y se remuestreó a "
            f"{config.SR} Hz, que es la frecuencia que usa XTTS-v2. Grabar "
            f"directamente a {config.SR} Hz evita esta conversión."
        )

    return validacion


def nombre_seguro(nombre: str | None) -> str:
    """
    Nombre de archivo reducido a su parte base, para mostrarlo sin riesgo.

    El nombre lo elige quien sube el archivo, así que puede traer rutas completas.
    Quedarse solo con la última parte evita mostrar la ruta del disco de otra
    persona y evita que un nombre como "../../algo" signifique nada.
    """
    if not nombre:
        return "audio.wav"
    # Las barras invertidas de Windows se pasan a barras normales para que
    # os.path.basename las reconozca también cuando el servidor corre en Linux.
    return os.path.basename(str(nombre).replace("\\", "/")) or "audio.wav"
