"""Ejecuta la etapa 2 desde terminal, sin levantar Flask (RNF-15).

    python -m clonvoz.cli_identidad audios/referencia.wav --consentimiento

La CLI usa el mismo codificador que la web. Siempre imprime un resumen y solo
escribe resultados cuando se pide --salida: exportar es una decisión explícita.
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from .audio import AudioInvalido
from .analisis import AudioRechazado
from .identidad import CodificadorIdentidad, ModeloNoDisponible, analizar_identidad, carpeta_modelo


def main(argv=None):
    """Pasos del comando:

    1. Leer ruta, dispositivo, consentimiento y destino opcional.
    2. Cargar los módulos oficiales y procesar la referencia.
    3. Imprimir formas, tiempos y estadísticas en JSON.
    4. Si se pidió exportar, guardar matrices, resumen y figuras.

    Devuelve 0 al terminar bien o 2 ante un problema previsto de entrada/modelo.
    argv=None usa los argumentos de la terminal; una lista permite probarlo.
    """
    # 1. argparse define las opciones y genera automáticamente la ayuda --help.
    parser = argparse.ArgumentParser(description="Etapa 2: Perceiver real de XTTS-v2, sin síntesis.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--modelo", type=Path, default=carpeta_modelo())
    parser.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--salida", type=Path, help="Exportar expresamente los tensores, resumen y gráficas")
    parser.add_argument("--consentimiento", action="store_true", help="Confirmar autorización para procesar esta voz")
    args = parser.parse_args(argv)
    if not args.consentimiento:
        parser.error("Confirma la autorización para usar esta voz en clonación académica con --consentimiento.")
    try:
        # 2. No se importa el servidor. El modelo y analizar_identidad se pueden
        # utilizar también desde otro programa Python sin interfaz gráfica.
        modelo = CodificadorIdentidad(args.modelo, args.dispositivo)
        resultado = analizar_identidad(args.audio, modelo)
    except (ModeloNoDisponible, AudioInvalido, AudioRechazado, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    # 3. ensure_ascii=False conserva los acentos; indent=2 lo hace legible.
    resumen = resultado.como_dict()
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    if args.salida:
        # 4. Importamos el render solo al exportar; imprimir cifras no necesita
        # dibujar ninguna imagen. mkdir crea también carpetas intermedias.
        from .graficas_identidad import renderizar_identidades
        args.salida.mkdir(parents=True, exist_ok=True)
        np.save(args.salida / "identidad.npy", resultado.vectores)
        # .npy conserva tipo y dimensiones. np.load recupera los números, mientras
        # que los PNG son solo su representación visual, no una copia del tensor.
        np.save(args.salida / "atencion.npy", resultado.atencion)
        np.save(args.salida / "atencion_consultas.npy", resultado.atencion_consultas)
        (args.salida / "identidad.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
        for nombre, temas in renderizar_identidades([resultado])[0].items():
            for tema, imagen in temas.items():
                # El render devuelve "data:image/png;base64,..." para la web.
                # Se quita la cabecera y se decodifica base64 para escribir un PNG.
                (args.salida / f"{nombre}_{tema}.png").write_bytes(base64.b64decode(imagen.split(",", 1)[1]))
    return 0


if __name__ == "__main__":
    # Solo se ejecuta al lanzar el comando, no al importar este archivo.
    raise SystemExit(main())
