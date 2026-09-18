"""
Las cuatro características visuales de la etapa 1.

Procesamiento de señales puro: estas funciones reciben y devuelven arreglos de
numpy, y no saben nada de matplotlib ni de Flask. Son las que la etapa 2 va a
importar directamente para alimentar al Perceiver Resampler, así que se mantienen
libres de cualquier dependencia de presentación.

    forma_de_onda        las muestras tal cual, con su eje de tiempo
    espectrograma_stft   [1 + N_FFT/2, T] en dB          (librosa)
    mel_espectrograma    [80, T] log-mel de XTTS-v2      (torchaudio)
    contorno_f0          [T] en Hz, con detección de voz (librosa)
"""

from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np

from . import config


def forma_de_onda(muestras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve `(tiempo, amplitud)` de la señal.

    No hay cálculo que hacer: la forma de onda *son* las muestras. Se acompaña
    del eje de tiempo en segundos para que la gráfica no se dibuje en índices.
    """
    amplitud = np.asarray(muestras, dtype=np.float32)
    tiempo = np.arange(amplitud.shape[0], dtype=np.float32) / config.SR
    return tiempo, amplitud


def espectrograma_stft(muestras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Espectrograma por Transformada de Fourier de Tiempo Corto, en decibelios.

    Devuelve `(espectrograma_db, frecuencias_hz)`, con el espectrograma de forma
    [1 + N_FFT/2, T]. Se usa el mismo N_FFT y el mismo salto que el mel, así que
    ambos mapas de calor comparten eje temporal y se pueden comparar de frente,
    que es justo lo que pide el documento de etapas.
    """
    stft = librosa.stft(
        np.asarray(muestras, dtype=np.float32),
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
    )
    espectrograma_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    frecuencias = librosa.fft_frequencies(sr=config.SR, n_fft=config.N_FFT)
    return espectrograma_db.astype(np.float32), frecuencias.astype(np.float32)


@lru_cache(maxsize=1)
def _transformada_mel():
    """
    Construye una sola vez el `MelSpectrogram` de torchaudio.

    El banco de filtros Mel se calcula al instanciar la transformada, no en cada
    llamada, así que rehacerlo por cada petición sería trabajo repetido y visible
    en los tiempos que reporta la interfaz.

    Los argumentos replican los que XTTS-v2 le pasa a `wav_to_mel_cloning` en el
    camino del Perceiver. Importante: `mel_scale` queda en "htk" (ver la nota en
    config.MEL_ESCALA).
    """
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SR,
        n_fft=config.N_FFT,
        win_length=config.WIN_LENGTH,
        hop_length=config.HOP_LENGTH,
        f_min=config.MEL_FMIN,
        f_max=config.MEL_FMAX,
        n_mels=config.N_MELS,
        power=config.MEL_POWER,
        normalized=False,
        norm=config.MEL_NORM,
        mel_scale=config.MEL_ESCALA,
    )


def mel_espectrograma(
    muestras: np.ndarray, estadisticas: np.ndarray | None = None
) -> np.ndarray:
    """
    Mel-espectrograma logarítmico de 80 canales, [80, T].

    Este es el tensor que la etapa 2 le entrega al Perceiver Resampler, y la razón
    por la que se calcula con torchaudio en vez de librosa: replica paso a paso lo
    que hace `wav_to_mel_cloning` de XTTS-v2, incluido el logaritmo con piso.

    `estadisticas` son las normas por canal que XTTS-v2 guarda en `mel_stats.pth`
    *dentro del checkpoint*, y por las que divide el mel al final. En la etapa 1
    el checkpoint todavía no se descarga, así que por omisión no se normaliza: la
    visualización no cambia, y la etapa 2 pasará el arreglo [80] cuando lo tenga.
    """
    import torch

    señal = torch.from_numpy(np.ascontiguousarray(muestras, dtype=np.float32))
    with torch.no_grad():
        mel = _transformada_mel()(señal)
        mel = torch.log(torch.clamp(mel, min=config.MEL_CLAMP_MIN))
        if estadisticas is not None:
            normas = torch.as_tensor(np.asarray(estadisticas), dtype=mel.dtype)
            mel = mel / normas.unsqueeze(-1)
    return mel.cpu().numpy()


def contorno_f0(
    muestras: np.ndarray, metodo: str = config.METODO_F0_POR_OMISION
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Contorno de frecuencia fundamental, [T] en Hz.

    Devuelve `(f0, voz_detectada, confianza)`. En los frames sin voz `f0` vale
    `nan` a propósito: al graficar, eso deja huecos en la línea en lugar de
    hundirla hasta cero, que es lo que hace ver entonaciones que no existen.

    El cálculo usa el mismo `frame_length` y el mismo salto que el mel, así que
    los T frames coinciden con los de las otras representaciones y el contorno se
    puede superponer directamente sobre la forma de onda.

    `metodo` puede ser "pyin" (preciso, detecta voz, lento) o "yin" (rápido, sin
    detección de voz).
    """
    señal = np.asarray(muestras, dtype=np.float32)

    if metodo == "yin":
        f0 = librosa.yin(
            señal,
            fmin=config.F0_MINIMO_HZ,
            fmax=config.F0_MAXIMO_HZ,
            sr=config.SR,
            frame_length=config.N_FFT,
            hop_length=config.HOP_LENGTH,
        )
        # yin no distingue frames sonoros de silencios: entrega un valor siempre.
        voz = np.ones_like(f0, dtype=bool)
        confianza = np.full_like(f0, np.nan, dtype=np.float32)
        return f0.astype(np.float32), voz, confianza

    if metodo != "pyin":
        raise ValueError(f"Método de F0 no reconocido: {metodo!r}. Usa 'pyin' o 'yin'.")

    f0, voz, confianza = librosa.pyin(
        señal,
        fmin=config.F0_MINIMO_HZ,
        fmax=config.F0_MAXIMO_HZ,
        sr=config.SR,
        frame_length=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    return (
        f0.astype(np.float32),
        np.asarray(voz, dtype=bool),
        np.asarray(confianza, dtype=np.float32),
    )


def eje_de_tiempo_frames(n_frames: int) -> np.ndarray:
    """Eje de tiempo en segundos para las representaciones por frames."""
    return librosa.frames_to_time(
        np.arange(n_frames), sr=config.SR, hop_length=config.HOP_LENGTH
    ).astype(np.float32)
