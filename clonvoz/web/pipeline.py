"""Rutas de la etapa 5: una corrida completa por HTTP y sus eventos por WebSocket.

    POST /api/inferencia            audio + texto -> 202 con el identificador
    WS   /ws/corrida/<id>           cada evento en el momento en que ocurre
    GET  /api/inferencia/<id>/audio el WAV final
    GET  /api/grabaciones           corridas grabadas de respaldo (etapa 6)
    GET  /api/grabaciones/<nombre>  una de ellas, para reproducirla sin modelo

El modelo corre en un hilo aparte y deja sus eventos en un registro; el canal
WebSocket los retransmite. Quien se conecta un instante tarde los recibe desde
el primero, y si el navegador se desconecta la corrida se detiene.
"""

import json
import os
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np
from flask import Blueprint, Response, current_app, jsonify, request, session
from flask_sock import Sock
from simple_websocket import ConnectionClosed

from .. import config
from ..analisis import AudioRechazado
from ..audio import AudioInvalido, nombre_seguro
from ..fases import a_wav
from ..generacion import GeneradorOcupado
from ..grabacion import Traductor, listar_grabaciones, ruta_de_grabacion
from ..identidad import ModeloNoDisponible
from ..pipeline import BLOQUES, CorridaCancelada, ejecutar_corrida
from .generacion import _fabrica_fases, fabrica_generador
from .identidad import fabrica_identidad
from .sintesis import fabrica_sintetizador

LIMITE_TEXTO = 239
METODOS_F0 = ("pyin", "yin")


class EstadoCorrida:
    """Los eventos de una corrida y lo necesario para esperarlos o detenerla."""

    def __init__(self, propietario):
        self.id = str(uuid.uuid4())
        self.propietario = propietario
        self.eventos = []
        self.condicion = threading.Condition()
        self.terminada = False
        self.detener = threading.Event()
        self.audio = None
        self.inicio = time.perf_counter()

    def emitir(self, tipo, datos):
        evento = {"tipo": tipo, "t": round((time.perf_counter() - self.inicio) * 1000), "datos": datos}
        with self.condicion:
            self.eventos.append(evento)
            self.condicion.notify_all()

    def terminar(self):
        with self.condicion:
            self.terminada = True
            self.condicion.notify_all()


class RegistroCorridas:
    """Las últimas corridas, por sesión. Solo una puede estar en curso: el modelo es uno."""

    def __init__(self, capacidad=4):
        self.capacidad = capacidad
        self.corridas = OrderedDict()
        self.cerrojo = threading.Lock()

    def crear(self, propietario):
        with self.cerrojo:
            if any(not c.terminada for c in self.corridas.values()):
                raise GeneradorOcupado("Ya hay una corrida en curso. Espera a que termine.")
            while len(self.corridas) >= self.capacidad:
                self.corridas.popitem(last=False)
            estado = EstadoCorrida(propietario)
            self.corridas[estado.id] = estado
            return estado

    def obtener(self, propietario, identificador):
        with self.cerrojo:
            estado = self.corridas.get(identificador)
        if estado is None or estado.propietario != propietario:
            raise KeyError(identificador)
        return estado


class ModelosDeCorrida:
    """Entrega al pipeline los mismos modelos, ya cargados, que usan las etapas sueltas."""

    def __init__(self, modelos):
        self.modelos = modelos

    def identidad(self):
        return self.modelos.obtener("identidad", fabrica_identidad)

    def generador(self):
        return self.modelos.obtener("generador", fabrica_generador)

    def fases(self):
        return self.modelos.obtener("fases", _fabrica_fases)

    def sintetizador(self):
        return self.modelos.obtener("sintetizador", fabrica_sintetizador)

    def reconocedor(self):
        return current_app.extensions["locutores"].reconocedor()


def carpeta_grabaciones():
    return os.environ.get("CLONVOZ_GRABACIONES", "grabaciones")


def crear_rutas_pipeline(app, memoria):
    """Crea las rutas de la etapa 5 sobre la memoria y los modelos compartidos."""
    rutas = Blueprint("pipeline", __name__)
    registro = RegistroCorridas()
    app.extensions["corridas"] = registro
    sock = Sock(app)

    def trabajar(estado, datos_audio, nombre, texto, opciones):
        """Ejecuta la corrida y traduce cada evento del pipeline a JSON para la página.

        Los objetos de cada etapa se guardan en la memoria de la sesión, igual que
        hacen las rutas sueltas: después de la corrida se puede seguir explorando.
        """
        propietario = estado.propietario
        traducir = Traductor(nombre, texto, estado.id)

        def anotar(**campos):
            try:
                memoria.anotar(propietario, estado.id, **campos)
            except KeyError:  # caducó o fue desalojada: no hay dónde anotar
                pass

        def emitir(tipo, datos):
            # Primero se guarda en la sesión; después se traduce al JSON de la página.
            if tipo == "etapa1":
                memoria.guardar(propietario, nombre, datos)
            elif tipo == "etapa2":
                anotar(identidad=datos.vectores)
            elif tipo == "fin":
                anotar(tokens_audio=np.asarray(datos["tokens"], dtype=np.int64),
                       texto_generado=texto.strip(), texto_ids=datos["texto_ids"], latentes=None)
            elif tipo == "etapa4":
                anotar(embedding_hablante=datos.embedding, latentes=datos.latentes)
                estado.audio = a_wav(datos.onda, sr=datos.sr)
            salida = traducir(tipo, datos)
            if tipo == "etapa1":
                salida["caducidad_s"] = memoria.caducidad_s
            estado.emitir(tipo, salida)

        try:
            ejecutar_corrida(datos_audio, texto, ModelosDeCorrida(current_app.extensions["modelos"]),
                             emitir, detener=estado.detener, id_corrida=estado.id, **opciones)
            estado.emitir("terminada", {"motivo": "completa"})
        except CorridaCancelada:
            estado.emitir("terminada", {"motivo": "cancelada"})
        except AudioRechazado as exc:
            estado.emitir("error", {"errores": list(exc.validacion.errores)})
        except (AudioInvalido, ModeloNoDisponible, GeneradorOcupado, ValueError) as exc:
            estado.emitir("error", {"errores": [str(exc)]})
        except Exception:  # el detalle queda en la terminal
            current_app.logger.exception("Error al ejecutar la corrida completa")
            estado.emitir("error", {"errores": ["No se pudo completar la corrida. Revisa la terminal."]})
        finally:
            estado.terminar()

    @rutas.post("/api/inferencia")
    def inferencia():
        """Lanza una corrida completa y responde de inmediato con su identificador.

        1. Validar consentimiento, archivo y texto (lo que no necesita el modelo).
        2. Registrar la corrida; si ya hay una en curso se rechaza.
        3. Arrancar el hilo de trabajo. Los eventos se leen por el WebSocket.
        """
        if request.form.get("consentimiento") != "si":
            return jsonify(errores=["Confirma la autorización para procesar esta voz."]), 400
        archivo = request.files.get("audio")
        datos_audio = archivo.read() if archivo is not None and archivo.filename else b""
        if not datos_audio:
            return jsonify(errores=["No llegó ningún archivo de audio en la petición."]), 400
        texto = (request.form.get("texto") or "").strip()
        if not texto:
            return jsonify(errores=["Escribe el texto que quieres generar."]), 400
        if len(texto) > LIMITE_TEXTO:
            return jsonify(errores=[f"El texto tiene {len(texto)} caracteres; el máximo en español "
                                    f"es {LIMITE_TEXTO}."]), 400
        metodo_f0 = request.form.get("metodo_f0", config.METODO_F0_POR_OMISION)
        if metodo_f0 not in METODOS_F0:
            return jsonify(errores=[f"Método de F0 no reconocido: {metodo_f0}."]), 400
        try:
            semilla = int(request.form.get("semilla", 0))
        except ValueError:
            semilla = 0
        opciones = {"determinista": request.form.get("determinista") == "si", "semilla": semilla,
                    "metodo_f0": metodo_f0, "explicar": request.form.get("explicar", "si") == "si"}

        propietario = session.get("propietario")
        if not propietario:
            return jsonify(errores=["Abre la página antes de lanzar una corrida."]), 400
        try:
            estado = registro.crear(propietario)
        except GeneradorOcupado as exc:
            return jsonify(errores=[str(exc)]), 409
        estado.emitir("corrida", {"id_corrida": estado.id, "texto": texto, **opciones,
                                  "bloques": [{"id": b, "nombre": n, "ruta": r} for b, n, r in BLOQUES]})

        aplicacion = current_app._get_current_object()
        nombre = nombre_seguro(archivo.filename)

        def en_contexto():
            with aplicacion.app_context():
                trabajar(estado, datos_audio, nombre, texto, opciones)

        threading.Thread(target=en_contexto, daemon=True).start()
        return jsonify(id_corrida=estado.id, ws=f"/ws/corrida/{estado.id}"), 202

    @rutas.get("/api/inferencia/<identificador>/audio")
    def audio_final(identificador):
        try:
            estado = registro.obtener(session.get("propietario"), identificador)
        except KeyError:
            return jsonify(errores=["Esa corrida no existe o ya no está disponible."]), 404
        if estado.audio is None:
            return jsonify(errores=["La corrida todavía no tiene audio."]), 409
        return Response(estado.audio, mimetype="audio/wav", headers={"Cache-Control": "no-store"})

    @rutas.get("/api/grabaciones")
    def grabaciones():
        """Las corridas completas guardadas en la carpeta de respaldo."""
        return jsonify(grabaciones=listar_grabaciones(carpeta_grabaciones()))

    @rutas.get("/api/grabaciones/<nombre>")
    def grabacion(nombre):
        try:
            ruta = ruta_de_grabacion(carpeta_grabaciones(), nombre)
        except KeyError:
            return jsonify(errores=["Esa grabación no existe."]), 404
        return Response(ruta.read_bytes(), mimetype="application/json", headers={"Cache-Control": "no-store"})

    @sock.route("/ws/corrida/<identificador>")
    def canal(ws, identificador):
        """Retransmite los eventos de la corrida; acepta {"accion": "detener"}."""
        try:
            estado = registro.obtener(session.get("propietario"), identificador)
        except KeyError:
            ws.send(json.dumps({"tipo": "error", "t": 0,
                                "datos": {"errores": ["Esa corrida no existe o ya no está disponible."]}}))
            return
        enviados = 0
        try:
            while True:
                with estado.condicion:
                    if enviados >= len(estado.eventos) and not estado.terminada:
                        estado.condicion.wait(timeout=0.25)
                    pendientes = estado.eventos[enviados:]
                    ultimo = estado.terminada
                for evento in pendientes:
                    ws.send(json.dumps(evento, ensure_ascii=False))
                enviados += len(pendientes)
                if ultimo and not pendientes:
                    break
                mensaje = ws.receive(timeout=0)
                if mensaje:
                    try:
                        if json.loads(mensaje).get("accion") == "detener":
                            estado.detener.set()
                    except (ValueError, AttributeError):
                        pass
        except ConnectionClosed:
            # Navegador desconectado: el GPT-2 se detiene en el siguiente paso.
            estado.detener.set()

    return rutas
