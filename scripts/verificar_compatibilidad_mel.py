"""Verificación de compatibilidad entre el mel calculado y wav_to_mel_cloning de XTTS-v2."""

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
    """Compara el mel calculado con la salida de wav_to_mel_cloning de coqui-tts."""
    try:
        import torch
        from TTS.tts.models.xtts import wav_to_mel_cloning
    except Exception as exc:
        print(f"  OMITIDA: no se pudo importar wav_to_mel_cloning de coqui-tts ({exc})")
        print("           Instala las dependencias del proyecto y vuelve a correr esto.")
        return None

    with torch.no_grad():
        señal = torch.from_numpy(muestras).unsqueeze(0)
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
    """Compara contra librosa evaluando diferencias de escala mel (htk vs slaney)."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Comprueba que el mel de la etapa 1 sea el que espera XTTS-v2."
    )
    parser.add_argument("audio", type=Path, help="Archivo .wav de referencia")
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
