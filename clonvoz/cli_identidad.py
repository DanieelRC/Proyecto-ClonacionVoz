"""Línea de comandos para extracción de identidad de voz mediante XTTS-v2."""

import argparse
import json
from pathlib import Path

import numpy as np

from .audio import AudioInvalido
from .analisis import AudioRechazado
from .identidad import CodificadorIdentidad, ModeloNoDisponible, analizar_identidad, carpeta_modelo


def main(argv=None):
    """Ejecuta el codificador de identidad sobre el audio y opcionalmente exporta tensores."""
    parser = argparse.ArgumentParser(description="Etapa 2: Perceiver Resampler de XTTS-v2 sin síntesis.")
    parser.add_argument("audio", type=Path, help="Archivo .wav de referencia")
    parser.add_argument("--modelo", type=Path, default=carpeta_modelo(), help="Ruta al checkpoint de XTTS-v2")
    parser.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto", help="Dispositivo de cómputo")
    parser.add_argument("--salida", type=Path, help="Carpeta de exportación de tensores y figuras")
    parser.add_argument("--consentimiento", action="store_true", help="Confirmar autorización para procesar esta voz")
    args = parser.parse_args(argv)

    if not args.consentimiento:
        parser.error("Confirma la autorización para usar esta voz en clonación académica con --consentimiento.")

    try:
        modelo = CodificadorIdentidad(args.modelo, args.dispositivo)
        resultado = analizar_identidad(args.audio, modelo)
    except (ModeloNoDisponible, AudioInvalido, AudioRechazado, ValueError) as exc:
        print(f"Error: {exc}")
        return 2

    resumen = resultado.como_dict()
    print(json.dumps(resumen, ensure_ascii=False, indent=2))

    if args.salida:
        from .graficas_identidad import renderizar_identidades
        args.salida.mkdir(parents=True, exist_ok=True)
        np.save(args.salida / "identidad.npy", resultado.vectores)
        np.save(args.salida / "atencion.npy", resultado.atencion)
        np.save(args.salida / "atencion_consultas.npy", resultado.atencion_consultas)
        (args.salida / "identidad.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

        for nombre, temas in renderizar_identidades([resultado])[0].items():
            for tema, imagen in temas.items():
                (args.salida / f"{nombre}_{tema}.png").write_bytes(imagen)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
