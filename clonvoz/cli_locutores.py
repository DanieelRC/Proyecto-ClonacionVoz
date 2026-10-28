"""Línea de comandos de la etapa 7: entrenar y probar el reconocimiento del locutor.

    python -m clonvoz.cli_locutores entrenar datos/locutores --consentimiento
    python -m clonvoz.cli_locutores identificar audio.m4a --consentimiento
"""

import argparse
import json
from pathlib import Path

from . import config
from .audio import AudioInvalido
from .identidad import CodificadorIdentidad, ModeloNoDisponible, carpeta_modelo
from .locutores import (ARCHIVO_EMBEDDINGS, ARCHIVO_MODELO, ARCHIVO_REPORTE, CARPETA_EXPORTACION,
                        cargar_reconocedor, carpeta_locutores, entrenar, extraer_conjunto, guardar,
                        leer_audio, validar_muestras)


def _entrenar(args):
    """1. Perceiver de cada audio (con caché). 2. Validación cruzada KNN/GMM. 3. Modelo y reporte a disco."""
    codificador = CodificadorIdentidad(args.modelo, args.dispositivo)
    limites = {"duracion_minima_s": args.duracion_minima, "duracion_maxima_s": args.duracion_maxima}

    def avance(indice, total, archivo):
        print(f"\r[{indice}/{total}] {archivo[:70]:<70}", end="", flush=True)

    conjunto = extraer_conjunto(args.datos, codificador, duracion_minima=args.duracion_minima,
                                duracion_maxima=args.duracion_maxima,
                                cache=None if args.sin_cache else args.salida / ARCHIVO_EMBEDDINGS,
                                al_avanzar=avance)
    print()
    for rechazado in conjunto.rechazados:
        print(f"  omitido: {rechazado['archivo']} ({rechazado['motivo']})")
    print(f"{len(conjunto.muestras)} audios de {len(conjunto.locutores)} locutores.")

    resultado = entrenar(conjunto, pliegues=args.pliegues, componentes=args.componentes,
                         dimensiones=args.dimensiones, aceptacion=args.aceptacion, semilla=args.semilla,
                         limites=limites, al_avanzar=print)
    reporte = resultado.reporte
    escritos = guardar(resultado, args.salida)

    print("\nBúsqueda del KNN (validación cruzada):")
    for fila in reporte["busqueda_knn"]:
        print(f"  k={fila['k']}  {fila['metrica']:<10}  exactitud {fila['exactitud']:.1%}  "
              f"recall {fila['recall_macro']:.1%}")
    print("\nComparación:")
    for clave, modelo in reporte["modelos"].items():
        marca = "  ← en uso" if clave == reporte["elegido"] else ""
        print(f"  {modelo['descripcion']:<44} exactitud {modelo['exactitud']:.1%}  "
              f"recall {modelo['recall_macro']:.1%}{marca}")
    print(f"\nModelo en uso: {reporte['motivo_eleccion']}")
    print(f"Modelo: {args.salida / ARCHIVO_MODELO}")
    print(f"Reporte: {args.salida / ARCHIVO_REPORTE}")
    print(f"Exportados ({len(escritos)}): {args.salida / CARPETA_EXPORTACION}")
    return 0


def _identificar(args):
    from .caracteristicas import mel_espectrograma

    reconocedor = cargar_reconocedor(args.salida)
    muestras = leer_audio(args.audio)
    motivo = validar_muestras(muestras, config.LOCUTOR_DURACION_MINIMA_S, config.LOCUTOR_DURACION_MAXIMA_S)
    if motivo:
        raise AudioInvalido(f"El audio {motivo}.")
    codificador = CodificadorIdentidad(args.modelo, args.dispositivo)
    vectores = codificador.procesar_mel(mel_espectrograma(muestras))[0]
    print(json.dumps(reconocedor.identificar(vectores).como_dict(), ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Etapa 7: reconocimiento del locutor con la salida del Perceiver.")
    parser.add_argument("--modelo", type=Path, default=carpeta_modelo(), help="Carpeta del checkpoint de XTTS-v2")
    parser.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--salida", type=Path, default=carpeta_locutores(),
                        help="Dónde guardar (o leer) el modelo y el reporte (por omisión modelos/locutores)")
    parser.add_argument("--consentimiento", action="store_true",
                        help="Confirmar la autorización de las personas para procesar sus voces")
    ordenes = parser.add_subparsers(dest="orden", required=True)

    entrenamiento = ordenes.add_parser("entrenar", help="Entrena KNN y GMM y elige el mejor")
    entrenamiento.add_argument("datos", type=Path, help="Carpeta con una subcarpeta de audios por locutor")
    entrenamiento.add_argument("--pliegues", type=int, default=config.LOCUTOR_PLIEGUES)
    entrenamiento.add_argument("--componentes", type=int, default=config.LOCUTOR_GMM_COMPONENTES,
                               help="Componentes de cada GMM")
    entrenamiento.add_argument("--dimensiones", type=int, default=config.LOCUTOR_GMM_DIMENSIONES,
                               help="Dimensiones del PCA antes del GMM")
    entrenamiento.add_argument("--aceptacion", type=float, default=config.LOCUTOR_ACEPTACION,
                               help="Fracción de aciertos que acepta el umbral de desconocido (0-1)")
    entrenamiento.add_argument("--duracion-minima", type=float, default=config.LOCUTOR_DURACION_MINIMA_S)
    entrenamiento.add_argument("--duracion-maxima", type=float, default=config.LOCUTOR_DURACION_MAXIMA_S)
    entrenamiento.add_argument("--semilla", type=int, default=0)
    entrenamiento.add_argument("--sin-cache", action="store_true",
                               help="Recalcular el Perceiver de todos los audios")

    identificacion = ordenes.add_parser("identificar", help="Identifica al locutor de un audio")
    identificacion.add_argument("audio", type=Path)
    args = parser.parse_args(argv)

    if not args.consentimiento:
        parser.error("Confirma la autorización de las personas para procesar sus voces con --consentimiento.")
    if args.orden == "entrenar" and not 0 < args.aceptacion <= 1:
        parser.error("--aceptacion debe estar entre 0 y 1.")

    try:
        return _entrenar(args) if args.orden == "entrenar" else _identificar(args)
    except (ModeloNoDisponible, AudioInvalido, FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
