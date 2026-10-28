"""Rutas de la etapa 7: reconocimiento del locutor.

    GET  /api/locutores                     reporte del entrenamiento con sus figuras
    GET  /api/locutores/exportar/<archivo>  un PNG, CSV o JSON exportado por el entrenamiento
    GET  /api/locutores/exportar.zip        todos juntos
    POST /api/locutor                       identifica la referencia ya pasada por la etapa 2

El entrenamiento no se hace aquí: lo hace `python -m clonvoz.cli_locutores`. Si
se vuelve a entrenar con el servidor abierto, el modelo y el reporte se releen
solos en la siguiente petición.
"""

import io
import re
import threading
import zipfile
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request, session

from ..grabacion import figuras_como_uri, respuesta_locutor
from ..graficas_locutores import renderizar_reporte
from ..identidad import ModeloNoDisponible
from ..locutores import (ARCHIVO_MODELO, ARCHIVO_REPORTE, CARPETA_EXPORTACION, cargar_reconocedor,
                         cargar_reporte, carpeta_locutores)

NOMBRE_EXPORTABLE = re.compile(r"^[\w\-.]{1,80}\.(png|csv|json)$")
TIPOS = {"png": "image/png", "csv": "text/csv; charset=utf-8", "json": "application/json"}


class AlmacenLocutores:
    """El reconocedor y el reporte en memoria; se recargan si el archivo cambió en disco."""

    def __init__(self, carpeta=None):
        self.carpeta = Path(carpeta or carpeta_locutores())
        self.cerrojo = threading.Lock()
        self._cache = {}

    def _version(self, nombre):
        ruta = self.carpeta / nombre
        return ruta.stat().st_mtime_ns if ruta.is_file() else None

    def _obtener(self, nombre, cargar):
        with self.cerrojo:
            version = self._version(nombre)
            guardado = self._cache.get(nombre)
            if guardado is None or guardado[0] != version:
                guardado = (version, cargar())
                self._cache[nombre] = guardado
            return guardado[1]

    def reconocedor(self):
        return self._obtener(ARCHIVO_MODELO, lambda: cargar_reconocedor(self.carpeta))

    def reporte(self):
        """El reporte con sus figuras ya dibujadas: dibujar 36×36 celdas tarda, se hace una vez."""
        def cargar():
            reporte = cargar_reporte(self.carpeta)
            return reporte, figuras_como_uri(renderizar_reporte(reporte))
        return self._obtener(ARCHIVO_REPORTE, cargar)

    def exportados(self):
        carpeta = self.carpeta / CARPETA_EXPORTACION
        if not carpeta.is_dir():
            return []
        return sorted(a.name for a in carpeta.iterdir() if a.is_file() and NOMBRE_EXPORTABLE.match(a.name))


def crear_rutas_locutores(app, memoria):
    rutas = Blueprint("locutores", __name__)
    almacen = AlmacenLocutores()
    app.extensions["locutores"] = almacen

    @rutas.get("/api/locutores")
    def reporte():
        try:
            datos, graficas = almacen.reporte()
        except ModeloNoDisponible as exc:
            return jsonify(hay_modelo=False, errores=[str(exc)])
        except (OSError, ValueError, KeyError) as exc:
            current_app.logger.exception("No se pudo leer el reporte de locutores")
            return jsonify(hay_modelo=False, errores=[f"El reporte está dañado: {exc}. Vuelve a entrenar."])
        salida = {k: v for k, v in datos.items() if k not in ("proyeccion", "predicciones")}
        salida.update(hay_modelo=True, graficas=graficas, exportados=almacen.exportados())
        return jsonify(salida)

    @rutas.get("/api/locutores/exportar/<nombre>")
    def exportado(nombre):
        if not NOMBRE_EXPORTABLE.match(nombre) or nombre not in almacen.exportados():
            return jsonify(errores=["Ese archivo no existe."]), 404
        ruta = almacen.carpeta / CARPETA_EXPORTACION / nombre
        return Response(ruta.read_bytes(), mimetype=TIPOS[nombre.rsplit(".", 1)[1]],
                        headers={"Content-Disposition": f'attachment; filename="{nombre}"',
                                 "Cache-Control": "no-store"})

    @rutas.get("/api/locutores/exportar.zip")
    def exportado_zip():
        nombres = almacen.exportados()
        if not nombres:
            return jsonify(errores=["Todavía no hay un reporte exportado."]), 404
        archivo = io.BytesIO()
        with zipfile.ZipFile(archivo, "w", zipfile.ZIP_DEFLATED) as comprimido:
            for nombre in nombres:
                comprimido.write(almacen.carpeta / CARPETA_EXPORTACION / nombre, nombre)
        return Response(archivo.getvalue(), mimetype="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="reporte_locutores.zip"',
                                 "Cache-Control": "no-store"})

    @rutas.post("/api/locutor")
    def identificar():
        """Usa la identidad [32, 1024] que la etapa 2 dejó en la memoria de la sesión."""
        datos = request.get_json(silent=True)
        if not isinstance(datos, dict) or datos.get("consentimiento") != "si":
            return jsonify(errores=["Confirma el consentimiento para procesar esta voz."]), 400
        try:
            entrada = memoria.obtener(session.get("propietario"), datos.get("id_corrida"))
        except KeyError:
            return jsonify(errores=["El análisis venció. Vuelve a ejecutar las etapas 1 y 2."]), 410
        if entrada.identidad is None:
            return jsonify(errores=["Primero ejecuta la etapa 2: el reconocedor usa su salida."]), 409
        try:
            identificacion = almacen.reconocedor().identificar(entrada.identidad)
        except ModeloNoDisponible as exc:
            return jsonify(errores=[str(exc)]), 503
        except ValueError as exc:
            return jsonify(errores=[str(exc)]), 400
        return jsonify(respuesta_locutor(identificacion, entrada.nombre))

    return rutas
