"""
Comprueba que el mel de la etapa 1 sea el mel que espera XTTS-v2.

    python scripts/verificar_compatibilidad_mel.py audios/muestra_hablante.wav

Por qué existe este script: en la etapa 2 el mel-espectrograma [80, T] que
calcula la etapa 1 se le entrega tal cual al Perceiver Resampler. Si algún
parámetro no coincide, el error no se manifiesta aquí sino allá, disfrazado de
"el modelo no funciona". Este script convierte esa suposición en un número medido.

Compara contra `wav_to_mel_cloning` de coqui-tts, que es la función que XTTS-v2
llama de verdad para alimentar al Perceiver, con los argumentos exactos que le
pasa `XTTS.get_gpt_cond_latents` en la rama del Perceiver.

Ojo si alguien decide "corregir" este script: la clase `TorchMelSpectrogram`
parece la candidata obvia y NO es la correcta (usa n_fft=1024 y sirve a Tortoise),
y los valores por omisión de la firma de `wav_to_mel_cloning` tampoco son los de
XTTS-v2 (son los del camino sin Perceiver, de XTTS-v1).

Devuelve 0 si todo coincide y 1 si algo se sale de la tolerancia, para poder
usarlo como verificación automática.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clonvoz import caracteristicas, config  # noqa: E402
from clonvoz.audio import cargar_audio  # noqa: E402

TOLERANCIA = 1e-4


def comparar_con_xtts(muestras: np.ndarray, nuestro_mel: np.ndarray) -> bool | None:
    """
    Compara contra la función real de XTTS-v2.

    Devuelve True/False si la comparación se pudo hacer, o None si coqui-tts no
    está instalado (en ese caso no se finge que pasó: se reporta como omitida).
    """
    try:
        import torch
        from TTS.tts.models.xtts import wav_to_mel_cloning
    except Exception as exc:
        print(f"  OMITIDA: no se pudo importar wav_to_mel_cloning de coqui-tts ({exc})")
        print("           Instala las dependencias del proyecto y vuelve a correr esto.")
        return None

    with torch.no_grad():
        # XTTS-v2 trabaja con un lote: [1, n] -> [1, 80, T].
        señal = torch.from_numpy(muestras).unsqueeze(0)
        # mel_norms de unos: la división por las estadísticas del checkpoint queda
        # neutralizada, para comparar el log-mel sin normalizar que es lo que
        # produce la etapa 1 mientras el checkpoint no esté descargado.
        suyo = wav_to_mel_cloning(
            señal,
            mel_norms=torch.ones(config.N_MELS),
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH,
            power=config.MEL_POWER,
            normalized=False,
            sample_rate=config.SR,
            f_min=config.MEL_FMIN,
            f_max=config.MEL_FMAX,
            n_mels=config.N_MELS,
        )
        suyo = suyo.squeeze(0).cpu().numpy()

    print(f"  forma nuestra          {list(nuestro_mel.shape)}")
    print(f"  forma de XTTS-v2       {list(suyo.shape)}")

    if suyo.shape != nuestro_mel.shape:
        print("  RESULTADO: FALLA — las formas no coinciden.")
        return False

    diferencia = float(np.max(np.abs(suyo - nuestro_mel)))
    print(f"  diferencia absoluta máxima  {diferencia:.3e}  (tolerancia {TOLERANCIA:.0e})")
    if diferencia <= TOLERANCIA:
        print("  RESULTADO: COINCIDE. El mel de la etapa 1 sirve tal cual en la etapa 2.")
        return True

    print("  RESULTADO: FALLA — revisa los parámetros en clonvoz/config.py.")
    return False


def comparar_con_librosa(muestras: np.ndarray, nuestro_mel: np.ndarray) -> None:
    """
    Compara contra librosa, igualando escala y normalización a propósito.

    No es una prueba que deba pasar, sino la evidencia de por qué la
    implementación usa torchaudio: se reporta también la diferencia que aparece
    al dejar la escala Mel de librosa en su valor por omisión (slaney), que es el
    error fácil de cometer.
    """
    import librosa

    def mel_librosa(htk: bool) -> np.ndarray:
        potencia = (
            np.abs(
                librosa.stft(
                    muestras,
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                    win_length=config.WIN_LENGTH,
                    pad_mode="reflect",
                )
            )
            ** 2
        )
        banco = librosa.filters.mel(
            sr=config.SR,
            n_fft=config.N_FFT,
            n_mels=config.N_MELS,
            fmin=config.MEL_FMIN,
            fmax=config.MEL_FMAX,
            htk=htk,
            norm="slaney",
        )
        return np.log(np.clip(banco @ potencia, config.MEL_CLAMP_MIN, None))

    igualado = mel_librosa(htk=True)
    por_omision = mel_librosa(htk=False)

    print(f"  librosa con htk=True (igualado)   dif. máx. {np.max(np.abs(igualado - nuestro_mel)):.3e}")
    print(f"  librosa con htk=False (omisión)   dif. máx. {np.max(np.abs(por_omision - nuestro_mel)):.3e}")
    print("  La segunda cifra es el tamaño del error que se evita usando torchaudio.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Comprueba que el mel de la etapa 1 sea el que espera XTTS-v2."
    )
    parser.add_argument("audio", type=Path, help="archivo .wav de referencia")
    args = parser.parse_args(argv)

    if not args.audio.is_file():
        print(f"error: no existe el archivo {args.audio}", file=sys.stderr)
        return 2

    audio = cargar_audio(args.audio)
    nuestro_mel = caracteristicas.mel_espectrograma(audio.muestras)

    print("\nParámetros en uso (clonvoz/config.py)")
    for clave in (
        "SR", "N_FFT", "HOP_LENGTH", "WIN_LENGTH", "N_MELS",
        "MEL_FMIN", "MEL_FMAX", "MEL_POWER", "MEL_NORM", "MEL_ESCALA", "MEL_CLAMP_MIN",
    ):
        print(f"  {clave:<15} {getattr(config, clave)}")

    print("\n1. Contra wav_to_mel_cloning, la función que usa XTTS-v2")
    resultado = comparar_con_xtts(audio.muestras, nuestro_mel)

    print("\n2. Contra librosa (referencia informativa)")
    comparar_con_librosa(audio.muestras, nuestro_mel)

    print()
    if resultado is False:
        return 1
    if resultado is None:
        print("Verificación incompleta: falta coqui-tts para la comparación que importa.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
