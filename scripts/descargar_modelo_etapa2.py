"""Prepara los dos archivos oficiales que necesita el cargador de etapa 2.

config.json describe dimensiones y opciones; model.pth contiene pesos aprendidos.
La descarga incluye el checkpoint completo porque así se distribuye. Después,
CodificadorIdentidad solo construye el encoder y el Perceiver y carga sus pesos.
"""

import argparse
from pathlib import Path
import urllib.request


def main():
    """Pasos: leer destino/uso, crear carpeta, descargar a .part y renombrar.

    No inicia inferencia ni descarga audios de personas. Ejecutar otra vez conserva
    los archivos definitivos existentes; no se comprueban nuevamente sus hashes.
    """
    # 1. Leer las opciones y exigir la confirmación del propósito académico.
    parser = argparse.ArgumentParser(description="Descargar checkpoint oficial de XTTS-v2 (aprox. 1.9 GB).")
    parser.add_argument("--destino", type=Path, default=Path("modelos/xtts_v2"))
    parser.add_argument("--uso-academico", action="store_true", help="Confirmar el uso académico conforme a CPML")
    args = parser.parse_args()
    if not args.uso_academico:
        parser.error("Lee https://huggingface.co/coqui/XTTS-v2 y confirma --uso-academico.")
    args.destino.mkdir(parents=True, exist_ok=True)
    # 2. parents crea las carpetas intermedias; exist_ok permite reutilizarlas.
    # Revisión fija: evita que otra descarga cambie los pesos silenciosamente.
    revision = "v2.0.2"
    for nombre in ("config.json", "model.pth"):
        destino = args.destino / nombre
        if destino.is_file():
            print(f"Ya existe {destino}; se conserva.", flush=True)
            continue
        url = f"https://huggingface.co/coqui/XTTS-v2/resolve/{revision}/{nombre}"
        parcial = destino.with_suffix(destino.suffix + ".part")
        # 3. Un nombre .part distingue una descarga incompleta de un peso listo.
        # Si falla la conexión, volver a ejecutar reinicia ese archivo parcial.
        print(f"Descargando {nombre}…", flush=True)
        # El archivo definitivo solo aparece cuando termina la descarga.
        with urllib.request.urlopen(url, timeout=120) as respuesta, parcial.open("wb") as archivo:
            # Leer 4 MB por vez evita almacenar todo el checkpoint en la RAM.
            # := asigna el bloque y comprueba si quedan bytes; b"" termina el bucle.
            while bloque := respuesta.read(4 * 1024 * 1024):
                archivo.write(bloque)
        parcial.replace(destino)
        # 4. Solo después de cerrar la descarga se publica el nombre definitivo.
        print(f"Listo: {destino}", flush=True)


if __name__ == "__main__":
    main()
