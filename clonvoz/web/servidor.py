"""
Servidor web de la etapa 1.

Dos rutas y nada más: la página y el análisis. El servidor no procesa señales por
su cuenta —eso vive en `clonvoz.analisis`, que también usa la línea de comandos—,
así que aquí solo hay traducción entre HTTP y el módulo de procesamiento.

**El audio no toca el disco en ningún momento** (RNF-13): llega en memoria, se
analiza en memoria y lo que se devuelve son imágenes ya renderizadas y números.
Al terminar la petición no queda rastro del audio en el servidor.
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from .. import config, graficas
from ..analisis import AudioRechazado, analizar
from ..audio import AudioInvalido, nombre_seguro

METODOS_F0 = ("pyin", "yin")


def crear_app() -> Flask:
    """
    Construye la aplicación Flask con sus rutas y sus límites.

    Es una función y no una variable global para poder crear una aplicación limpia
    en cada prueba, sin que el estado de una se filtre a la siguiente.
    """
    app = Flask(__name__)

    # Flask corta por su cuenta cualquier petición más pesada que esto y levanta
    # RequestEntityTooLarge, que se atiende más abajo. Protege de que una carga
    # enorme llene la memoria del servidor antes de poder rechazarla.
    app.config["MAX_CONTENT_LENGTH"] = config.TAMANO_MAXIMO_BYTES

    # Sin esto Flask ordena las claves del JSON alfabéticamente y el orden pensado
    # —metadatos, formas, estadísticas, tiempos— se pierde al leerlo a mano.
    app.json.sort_keys = False

    @app.get("/")
    def pagina():
        # Los límites se pasan a la plantilla para que la validación del
        # navegador y la del servidor salgan del mismo config.py y no se
        # desincronicen.
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
        """
        Recibe el .wav, lo analiza y devuelve números e imágenes.

        La receta, paso a paso:
          1. Comprobar que venga un archivo y que el método de F0 sea conocido.
          2. Leer los bytes del archivo, sin guardarlos en ningún lado.
          3. Analizar, traduciendo cada fallo a un 400 con su explicación.
          4. Dibujar las ocho imágenes y empaquetarlas con el resumen.
        """
        # 1. request.files trae los archivos del formulario. Se comprueba también el
        #    nombre porque un campo vacío llega igualmente como archivo.
        archivo = request.files.get("audio")
        if archivo is None or not archivo.filename:
            return _error("No llegó ningún archivo de audio en la petición."), 400

        # El método lo elige el navegador, así que se valida contra la lista en vez
        # de pasarlo directo al procesamiento.
        metodo_f0 = request.form.get("metodo_f0", config.METODO_F0_POR_OMISION)
        if metodo_f0 not in METODOS_F0:
            return _error(f"Método de F0 no reconocido: {metodo_f0}."), 400

        # 2. Los bytes crudos del .wav, en memoria. De aquí salen hacia `analizar`,
        #    que sabe leerlos sin necesidad de un archivo en disco.
        datos = archivo.read()
        if not datos:
            return _error("El archivo de audio llegó vacío."), 400

        # 3. AudioInvalido es "no se pudo leer" y AudioRechazado es "se leyó pero no
        #    cumple". El segundo trae varios motivos y se envían todos juntos, para
        #    no obligar a corregir de uno en uno.
        try:
            analisis = analizar(datos, metodo_f0=metodo_f0)
        except AudioInvalido as exc:
            return _error(str(exc)), 400
        except AudioRechazado as exc:
            return _error(*exc.validacion.errores), 400

        # 4. El resumen en números (formas, estadísticas, tiempos) más el nombre del
        #    archivo, ya reducido a su parte base por seguridad.
        respuesta = analisis.como_dict()
        respuesta["archivo"] = nombre_seguro(archivo.filename)

        # Las cuatro figuras en sus dos temas: ocho PNG, convertidos a texto para
        # que quepan dentro del JSON y el navegador los muestre sin otra petición.
        imagenes = graficas.renderizar(analisis)
        respuesta["graficas"] = {
            nombre: {tema: graficas.como_data_uri(png) for tema, png in por_tema.items()}
            for nombre, por_tema in imagenes.items()
        }
        return jsonify(respuesta)

    @app.errorhandler(RequestEntityTooLarge)
    def demasiado_grande(_error_original):
        # Sin este manejador, pasarse del tamaño devolvería la página de error de
        # Flask en HTML y el navegador fallaría al intentar leerla como JSON.
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
        # El traceback queda en la consola del servidor; al navegador va un
        # mensaje en español, nunca la página de error de Flask.
        return _error("Ocurrió un error inesperado al analizar el audio."), 500

    return app


def _error(*mensajes: str):
    """
    Respuesta de error uniforme: siempre JSON, siempre en español.

    Recibe varios mensajes porque la validación puede encontrar más de un motivo, y
    el navegador espera siempre la misma forma: una lista bajo la clave "errores".
    """
    return jsonify({"errores": list(mensajes)})
