"""
Línea de comandos de la etapa 1.

    python -m clonvoz.cli audios/muestra_hablante.wav --salida salidas/ --guardar-mel

Este módulo existe para cumplir RNF-15: el procesamiento debe poder probarse por
línea de comandos, sin levantar la aplicación web. No importa Flask en ningún
punto, y esa independencia es justamente lo que hay que poder demostrar.

También es el puente concreto con la etapa 2: con `--guardar-mel` deja un
`mel.npy` de [80, T] que el Perceiver Resampler podrá cargar sin volver a
procesar el audio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import config, graficas
from .analisis import AudioRechazado, analizar
from .audio import AudioInvalido

NOMBRES_DE_ARCHIVO = {
    "forma_de_onda": "01_forma_de_onda.png",
    "espectrograma": "02_espectrograma_stft.png",
    "mel": "03_mel_espectrograma.png",
    "f0": "04_contorno_f0.png",
}


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m clonvoz.cli",
        description=(
            "Etapa 1: extrae de un .wav la forma de onda, el espectrograma STFT, "
            "el mel-espectrograma de 80 canales y el contorno de F0, y guarda las "
            "cuatro visualizaciones."
        ),
    )
    parser.add_argument("audio", type=Path, help="archivo .wav de entrada")
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("salidas"),
        help="carpeta donde escribir los resultados (por omisión: salidas/)",
    )
    parser.add_argument(
        "--tema",
        choices=("claro", "oscuro", "ambos"),
        default="claro",
        help="tema de las figuras (por omisión: claro)",
    )
    parser.add_argument(
        "--metodo-f0",
        choices=("pyin", "yin"),
        default=config.METODO_F0_POR_OMISION,
        help=(
            "estimador de F0: pyin es más preciso y detecta voz, yin es más "
            f"rápido (por omisión: {config.METODO_F0_POR_OMISION})"
        ),
    )
    parser.add_argument(
        "--guardar-mel",
        action="store_true",
        help="además, guardar mel.npy [80, T] para usarlo como entrada de la etapa 2",
    )
    return parser


def _imprimir_resumen(analisis, destino: Path) -> None:
    meta = analisis.metadatos
    est = analisis.estadisticas

    print(f"\nCorrida {analisis.id_corrida}")
    print(f"  archivo        {analisis.audio.formato} {analisis.audio.subtipo}")
    print(f"  duración       {meta['duracion_s']:.2f} s")
    print(f"  muestreo       {meta['sr_original']} Hz -> {meta['sr_trabajo']} Hz")
    print(f"  canales        {meta['canales_original']} -> 1 (mono)")

    print("\n  Tensores")
    for nombre, forma in analisis.formas.items():
        print(f"    {nombre:<16} {forma}")

    print("\n  Estadísticas")
    print(f"    RMS / pico     {est['rms']:.5f} / {est['pico']:.3f}")
    print(f"    mel (min/max)  {est['mel_min']:.2f} / {est['mel_max']:.2f}")
    if est["f0_media_hz"] is not None:
        print(
            f"    F0             media {est['f0_media_hz']:.0f} Hz, "
            f"rango {est['f0_min_hz']:.0f}-{est['f0_max_hz']:.0f} Hz"
        )
    else:
        print("    F0             no se detectó voz")
    print(f"    frames con voz {est['porcentaje_con_voz']:.1f} %")

    print("\n  Tiempos (ms)")
    for paso, ms in analisis.tiempos_ms.items():
        print(f"    {paso:<16} {ms:8.1f}")

    for advertencia in analisis.validacion.advertencias:
        print(f"\n  Advertencia: {advertencia}")

    print(f"\n  Resultados en {destino.resolve()}\n")


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)

    if not args.audio.is_file():
        print(f"error: no existe el archivo {args.audio}", file=sys.stderr)
        return 2

    try:
        analisis = analizar(args.audio, metodo_f0=args.metodo_f0)
    except AudioInvalido as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except AudioRechazado as exc:
        print("error: el audio no cumple los requisitos:", file=sys.stderr)
        for motivo in exc.validacion.errores:
            print(f"  - {motivo}", file=sys.stderr)
        return 2

    temas = ("claro", "oscuro") if args.tema == "ambos" else (args.tema,)
    imagenes = graficas.renderizar(analisis, temas=temas)

    args.salida.mkdir(parents=True, exist_ok=True)
    for figura, por_tema in imagenes.items():
        for tema, png in por_tema.items():
            nombre = NOMBRES_DE_ARCHIVO[figura]
            if len(temas) > 1:
                nombre = nombre.replace(".png", f"_{tema}.png")
            (args.salida / nombre).write_bytes(png)

    resumen = analisis.como_dict()
    (args.salida / "caracteristicas.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.guardar_mel:
        np.save(args.salida / "mel.npy", analisis.mel.astype(np.float32))

    _imprimir_resumen(analisis, args.salida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
