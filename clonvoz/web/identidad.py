"""Rutas web para la extracción y visualización de identidad (etapa 2)."""

import os
import threading
import time

from flask import Blueprint, current_app, jsonify, request, session

from ..identidad import CodificadorIdentidad, ModeloNoDisponible, identidad_desde_mel, carpeta_modelo
from ..graficas_identidad import renderizar_identidades
from ..graficas import como_data_uri


def crear_rutas_identidad(memoria):
    """Crea el blueprint con los endpoints de la etapa 2 utilizando la memoria de mel compartida."""
    rutas = Blueprint("identidad", __name__)
    modelo = None
    cerrojo = threading.Lock()

    @rutas.post("/api/descartar")
    def descartar():
        """Libera de la memoria el mel de una corrida previa."""
        datos = request.get_json(silent=True) or {}
        if not isinstance(datos, dict) or not isinstance(datos.get("id_corrida"), str):
            return jsonify(errores=["Falta el identificador del análisis."]), 400
        memoria.descartar(session.get("propietario"), datos["id_corrida"])
        return jsonify(descartado=True)

    @rutas.post("/api/identidad")
    def identidad():
        """Ejecuta el Perceiver sobre los mels guardados en sesión y devuelve resultados y visualizaciones."""
        nonlocal modelo
        datos = request.get_json(silent=True)
        if not isinstance(datos, dict) or datos.get("consentimiento") != "si":
            return jsonify(errores=["Completa la etapa 1 y confirma el consentimiento."]), 400

        ids = [datos.get("id_corrida")]
        if datos.get("comparacion_id") is not None:
            ids.append(datos["comparacion_id"])

        if not all(isinstance(i, str) and i for i in ids):
            return jsonify(errores=["Primero ejecuta la etapa 1 para esta referencia."]), 400

        try:
            entradas = [memoria.obtener(session.get("propietario"), i) for i in ids]
        except KeyError:
            return jsonify(errores=["El resultado de etapa 1 venció o ya no está disponible. Vuelve a analizar la referencia en etapa 1."]), 410

        try:
            inicio = time.perf_counter()
            with cerrojo:
                if modelo is None:
                    modelo = CodificadorIdentidad(carpeta_modelo(), os.environ.get("CLONVOZ_DISPOSITIVO", "auto"))
            carga_ms = (time.perf_counter() - inicio) * 1000

            resultados = [identidad_desde_mel(e.mel, modelo, e.id_corrida, e.duracion_s, e.advertencias)
                          for e in entradas]
            figuras = renderizar_identidades(resultados)

            salida = []
            for entrada, resultado, graficas in zip(entradas, resultados, figuras):
                resumen = resultado.como_dict()
                resumen["archivo"] = entrada.nombre
                resumen["graficas"] = {vista: {tema: como_data_uri(png) for tema, png in temas.items()}
                                       for vista, temas in graficas.items()}
                salida.append(resumen)

            return jsonify(resultados=salida, carga_modelo_ms=round(carga_ms, 2))
        except ModeloNoDisponible as exc:
            return jsonify(errores=[str(exc)]), 503
        except ValueError as exc:
            return jsonify(errores=[str(exc)]), 400
        except Exception:
            current_app.logger.exception("Error al ejecutar la etapa 2")
            return jsonify(errores=["No se pudo completar la etapa 2. Revisa el detalle en la terminal."]), 500

    return rutas
