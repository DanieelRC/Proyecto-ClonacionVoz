"""Extracción de características acústicas de audio (forma de onda, STFT, mel-espectrograma y F0)."""

from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np

from . import config


def forma_de_onda(muestras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve el vector de tiempo en segundos y la amplitud de la señal."""
    amplitud = np.asarray(muestras, dtype=np.float32)
    tiempo = np.arange(amplitud.shape[0], dtype=np.float32) / config.SR
    return tiempo, amplitud


def espectrograma_stft(muestras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula el espectrograma STFT en decibelios [1 + N_FFT/2, T] y las frecuencias en Hz."""
    stft = librosa.stft(
        np.asarray(muestras, dtype=np.float32),
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
    )
    # Magnitud a escala logarítmica relativa al valor pico
    espectrograma_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    frecuencias = librosa.fft_frequencies(sr=config.SR, n_fft=config.N_FFT)
    return espectrograma_db.astype(np.float32), frecuencias.astype(np.float32)


@lru_cache(maxsize=1)
def _transformada_mel():
    """Instancia y almacena en caché la transformada MelSpectrogram de torchaudio."""
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
    """Calcula el mel-espectrograma logarítmico [80, T] con normalización opcional."""
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
    """Estima el contorno de frecuencia fundamental F0 en Hz mediante 'pyin' o 'yin'."""
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
    """Calcula el eje de tiempo en segundos para las representaciones basadas en frames."""
    return librosa.frames_to_time(
        np.arange(n_frames), sr=config.SR, hop_length=config.HOP_LENGTH
    ).astype(np.float32)
