"""Servidor web Flask para la interfaz y API de análisis acústico."""

from __future__ import annotations

import secrets

from flask import Flask, jsonify, render_template, request, session
from werkzeug.exceptions import RequestEntityTooLarge

from .. import config, graficas
from ..analisis import AudioRechazado, analizar
from ..audio import AudioInvalido, nombre_seguro

METODOS_F0 = ("pyin", "yin")


def crear_app() -> Flask:
    """Crea y configura la aplicación Flask con sus rutas y extensiones."""
    app = Flask(__name__)
    app.secret_key = secrets.token_bytes(32)
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    from .memoria import MemoriaMel
    memoria = MemoriaMel()
    app.extensions["memoria_mel"] = memoria

    from .identidad import crear_rutas_identidad
    app.register_blueprint(crear_rutas_identidad(memoria))

    app.config["MAX_CONTENT_LENGTH"] = config.TAMANO_MAXIMO_BYTES
    app.json.sort_keys = False

    @app.get("/")
    def pagina():
        session.setdefault("propietario", secrets.token_urlsafe(32))
        return render_template(
            "index.html",
            sr=config.SR,
            n_mels=config.N_MELS,
            duracion_minima=config.DURACION_MINIMA_S,
            duracion_maxima=config.DURACION_MAXIMA_S,
            duracion_recomendada_minima=config.DURACION_RECOMENDADA_MINIMA_S,
            duracion_recomendada_maxima=config.DURACION_RECOMENDADA_MAXIMA_S,
            tamano_maximo_mb=config.TAMANO_MAXIMO_BYTES // (1024 * 1024),
        )

    @app.post("/api/analizar")
    def api_analizar():
        """Recibe el archivo de audio, ejecuta el análisis y devuelve métricas e imágenes."""
        if request.form.get("consentimiento") != "si":
            return _error("Confirma la autorización para procesar esta voz."), 400

        archivo = request.files.get("audio")
        if archivo is None or not archivo.filename:
            return _error("No llegó ningún archivo de audio en la petición."), 400

        metodo_f0 = request.form.get("metodo_f0", config.METODO_F0_POR_OMISION)
        if metodo_f0 not in METODOS_F0:
            return _error(f"Método de F0 no reconocido: {metodo_f0}."), 400

        datos = archivo.read()
        if not datos:
            return _error("El archivo de audio llegó vacío."), 400

        try:
            analisis = analizar(datos, metodo_f0=metodo_f0)
        except AudioInvalido as exc:
            return _error(str(exc)), 400
        except AudioRechazado as exc:
            return _error(*exc.validacion.errores), 400

        respuesta = analisis.como_dict()
        respuesta["archivo"] = nombre_seguro(archivo.filename)

        imagenes = graficas.renderizar(analisis)
        respuesta["graficas"] = {
            nombre: {tema: graficas.como_data_uri(png) for tema, png in por_tema.items()}
            for nombre, por_tema in imagenes.items()
        }

        propietario = session.setdefault("propietario", secrets.token_urlsafe(32))
        memoria.guardar(propietario, respuesta["archivo"], analisis)
        respuesta["caducidad_s"] = memoria.caducidad_s
        return jsonify(respuesta)

    @app.errorhandler(RequestEntityTooLarge)
    def demasiado_grande(_error_original):
        mb = config.TAMANO_MAXIMO_BYTES // (1024 * 1024)
        return (
            _error(
                f"El archivo pesa más de {mb} MB. Sube una muestra más corta: "
                f"con {config.DURACION_MAXIMA_S:.0f} s de audio basta."
            ),
            413,
        )

    @app.errorhandler(500)
    def error_interno(_error_original):
        return _error("Ocurrió un error inesperado al analizar el audio."), 500

    return app


def _error(*mensajes: str):
    """Genera una respuesta de error estandarizada en formato JSON."""
    return jsonify({"errores": list(mensajes)})
