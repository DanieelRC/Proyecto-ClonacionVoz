"""Descarga de los archivos de configuración y pesos oficiales de XTTS-v2."""

import argparse
from pathlib import Path
import urllib.request


def main():
    """Descarga config.json y model.pth de Hugging Face en la carpeta de destino."""
    parser = argparse.ArgumentParser(description="Descargar checkpoint oficial de XTTS-v2 (~1.9 GB).")
    parser.add_argument("--destino", type=Path, default=Path("modelos/xtts_v2"), help="Carpeta de destino")
    parser.add_argument("--uso-academico", action="store_true", help="Confirmar el uso académico conforme a CPML")
    args = parser.parse_args()

    if not args.uso_academico:
        parser.error("Lee https://huggingface.co/coqui/XTTS-v2 y confirma --uso-academico.")

    args.destino.mkdir(parents=True, exist_ok=True)
    revision = "v2.0.2"

    for nombre in ("config.json", "model.pth"):
        destino = args.destino / nombre
        if destino.is_file():
            print(f"Ya existe {destino}; se conserva.", flush=True)
            continue

        url = f"https://huggingface.co/coqui/XTTS-v2/resolve/{revision}/{nombre}"
        parcial = destino.with_suffix(destino.suffix + ".part")
        print(f"Descargando {nombre}…", flush=True)

        with urllib.request.urlopen(url, timeout=120) as respuesta, parcial.open("wb") as archivo:
            while bloque := respuesta.read(4 * 1024 * 1024):
                archivo.write(bloque)
        parcial.replace(destino)
        print(f"Listo: {destino}", flush=True)


if __name__ == "__main__":
    main()
