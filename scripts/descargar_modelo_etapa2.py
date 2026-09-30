"""Descarga de los archivos de configuración y pesos oficiales de XTTS-v2."""

import argparse
from pathlib import Path
import urllib.error
import urllib.request

REVISION = "v2.0.2"
ARCHIVOS = ("config.json", "model.pth", "vocab.json", "dvae.pth")
"""vocab.json lo usa el tokenizador de la etapa 3; dvae.pth, solo las fases audibles 3 y 4."""


def descargar(nombre, destino):
    """Baja un archivo de la revisión fijada; el DVAE, si esa revisión no lo trae, de main."""
    revisiones = (REVISION, "main") if nombre == "dvae.pth" else (REVISION,)
    for revision in revisiones:
        url = f"https://huggingface.co/coqui/XTTS-v2/resolve/{revision}/{nombre}"
        parcial = destino.with_suffix(destino.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as respuesta, parcial.open("wb") as archivo:
                while bloque := respuesta.read(4 * 1024 * 1024):
                    archivo.write(bloque)
        except urllib.error.HTTPError as exc:
            parcial.unlink(missing_ok=True)
            if exc.code == 404 and revision != revisiones[-1]:
                continue
            raise
        parcial.replace(destino)
        return revision


def main():
    """Descarga config.json, model.pth, vocab.json y dvae.pth en la carpeta de destino."""
    parser = argparse.ArgumentParser(description="Descargar checkpoint oficial de XTTS-v2 (~2.1 GB).")
    parser.add_argument("--destino", type=Path, default=Path("modelos/xtts_v2"), help="Carpeta de destino")
    parser.add_argument("--uso-academico", action="store_true", help="Confirmar el uso académico conforme a CPML")
    args = parser.parse_args()

    if not args.uso_academico:
        parser.error("Lee https://huggingface.co/coqui/XTTS-v2 y confirma --uso-academico.")

    args.destino.mkdir(parents=True, exist_ok=True)

    for nombre in ARCHIVOS:
        destino = args.destino / nombre
        if destino.is_file():
            print(f"Ya existe {destino}; se conserva.", flush=True)
            continue
        print(f"Descargando {nombre}…", flush=True)
        revision = descargar(nombre, destino)
        print(f"Listo: {destino} (revisión {revision})", flush=True)


if __name__ == "__main__":
    main()
