"""Etapa 2 consume el mel guardado por etapa 1; nunca recibe otro WAV.

El navegador envía identificadores. El servidor recupera las matrices de su
sesión y las entrega al codificador sin volver a cargar o analizar audio.
"""

import os
import threading
import time

from flask import Blueprint, current_app, jsonify, request, session

from ..identidad import CodificadorIdentidad, ModeloNoDisponible, identidad_desde_mel, carpeta_modelo
from ..graficas_identidad import renderizar_identidades


def crear_rutas_identidad(memoria):
    """Comparte con etapa 1 la memoria temporal y carga el modelo bajo demanda."""
    rutas = Blueprint("identidad", __name__)
    modelo = None
    cerrojo = threading.Lock()

    @rutas.post("/api/descartar")
    def descartar():
        """Libera un mel cuando se cambia la referencia o se quita la comparación."""
        datos = request.get_json(silent=True) or {}
        if not isinstance(datos, dict) or not isinstance(datos.get("id_corrida"), str):
            return jsonify(errores=["Falta el identificador del análisis."]), 400
        memoria.descartar(session.get("propietario"), datos["id_corrida"])
        return jsonify(descartado=True)

    @rutas.post("/api/identidad")
    def identidad():
        """Pasos: validar ids -> recuperar mels -> codificar -> dibujar resultados.

        La comprobación de sesión impide usar un resultado de otro navegador.
        No se permite saltar etapa 1 enviando archivos a este endpoint.
        """
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
            # Solo se inicializan los pesos una vez; los mels ya están calculados.
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
                resumen["graficas"] = graficas
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
