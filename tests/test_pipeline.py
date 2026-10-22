"""Pruebas de la etapa 5: el pipeline único y su transmisión por WebSocket.

El servidor se levanta de verdad en un puerto libre, porque el cliente de pruebas
de Flask no habla WebSocket.
"""

import http.cookiejar
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import uuid
import warnings
from pathlib import Path

try:
    import torch  # noqa: F401
    import TTS  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Las pruebas de la etapa 5 necesitan PyTorch y coqui-tts.")

from simple_websocket import Client, ConnectionClosed
from werkzeug.serving import make_server

from clonvoz.fases import ReconstructorFases
from clonvoz.grabacion import FORMATO, Grabador, Traductor, ruta_de_grabacion
from clonvoz.identidad import ModeloNoDisponible
from clonvoz.pipeline import BLOQUES, CorridaCancelada, ejecutar_corrida
from clonvoz.web.servidor import crear_app
from test_generacion import ModeloIdentidadChico, dvae_aleatorio, generador_chico, voz_prueba, wav
from test_sintesis import sintetizador_chico

warnings.filterwarnings("ignore")


class ModelosChicos:
    """Los modelos reducidos, entregados como los pide el pipeline."""

    def __init__(self, carpeta_dvae=None):
        self._generador = generador_chico()
        self._sintetizador = sintetizador_chico(self._generador)
        self._fases = (ReconstructorFases(carpeta_dvae, ModeloIdentidadChico.normas)
                       if carpeta_dvae else None)

    def identidad(self):
        return ModeloIdentidadChico()

    def generador(self):
        return self._generador

    def sintetizador(self):
        return self._sintetizador

    def fases(self):
        if self._fases is None:
            raise ModeloNoDisponible("Falta dvae.pth")
        return self._fases


class PruebasPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.carpeta = tempfile.TemporaryDirectory()
        dvae_aleatorio(cls.carpeta.name)
        cls.voz = wav(voz_prueba(3.0))

    @classmethod
    def tearDownClass(cls):
        cls.carpeta.cleanup()

    def correr(self, modelos, **opciones):
        eventos = []
        corrida = ejecutar_corrida(self.voz, "Hola mundo", modelos, lambda t, d: eventos.append((t, d)),
                                   determinista=True, metodo_f0="yin", **opciones)
        return corrida, eventos

    def test_corrida_completa_en_orden(self):
        corrida, eventos = self.correr(ModelosChicos(self.carpeta.name))
        bloques = [d for t, d in eventos if t == "bloque"]
        # Cada bloque se activa antes de terminar, y terminan en el orden del diagrama.
        self.assertEqual([b["id"] for b in bloques if b["estado"] == "listo"], [b[0] for b in BLOQUES])
        for identificador, _, _ in BLOQUES:
            estados = [b["estado"] for b in bloques if b["id"] == identificador]
            self.assertEqual(estados[0], "activo")
            self.assertEqual(estados[-1], "listo")
        formas = {b["id"]: b["forma"] for b in bloques if b["estado"] == "listo"}
        self.assertEqual(formas["mel"][0], 80)
        self.assertEqual(formas["perceiver"][0], 32)
        self.assertEqual(formas["texto"], [len(corrida.generacion.texto_ids)])
        self.assertEqual(formas["hablante"], [512])
        self.assertEqual(formas["tokens"], [len(corrida.generacion.tokens)])
        self.assertEqual(formas["salida"], [len(corrida.sintesis.onda)])

        tipos = [t for t, _ in eventos if t != "bloque"]
        pasos = tipos.count("paso")
        self.assertEqual(tipos, ["etapa1", "etapa2", "inicio"] + ["paso"] * pasos +
                         ["fin", "etapa4", "explicando", "causal"])
        self.assertEqual(pasos, len(corrida.generacion.tokens))
        self.assertEqual(corrida.causal["pasos"], pasos)
        self.assertIsNotNone(corrida.sintesis.mel_dvae)
        # Los eventos pequeños son JSON tal cual.
        json.dumps([d for t, d in eventos if t in ("bloque", "inicio", "paso", "fin", "causal")],
                   allow_nan=False)

    def test_grabacion_reproduce_los_eventos_de_la_web(self):
        grabador = Grabador()
        traducir = Traductor("voz.wav", "Hola mundo", "grabada")
        ejecutar_corrida(self.voz, "Hola mundo", ModelosChicos(self.carpeta.name),
                         lambda t, d: grabador(t, traducir(t, d)), determinista=True, metodo_f0="yin",
                         id_corrida="grabada")
        corrida = json.loads(json.dumps(grabador.como_dict(), allow_nan=False))
        self.assertEqual(corrida["formato"], FORMATO)
        porTipo = {e["tipo"]: e["datos"] for e in corrida["eventos"]}
        self.assertEqual(porTipo["etapa1"]["id_corrida"], "grabada")
        self.assertIn("mel", porTipo["etapa1"]["graficas"])
        self.assertIn("atencion", porTipo["etapa2"]["resultados"][0]["graficas"])
        self.assertTrue(porTipo["etapa4"]["audio"].startswith("data:audio/wav;base64,"))
        tiempos = [e["t"] for e in corrida["eventos"]]
        self.assertEqual(tiempos, sorted(tiempos))

    def test_nombres_de_grabacion(self):
        with tempfile.TemporaryDirectory() as carpeta:
            (Path(carpeta) / "respaldo.json").write_text("{}", encoding="utf-8")
            self.assertEqual(ruta_de_grabacion(carpeta, "respaldo.json").name, "respaldo.json")
            for nombre in ("../secreto.json", "..\\x.json", "otro.txt", ".oculto.json", "falta.json", ""):
                with self.assertRaises(KeyError):
                    ruta_de_grabacion(carpeta, nombre)

    def test_sin_dvae_la_corrida_sigue(self):
        corrida, eventos = self.correr(ModelosChicos(), explicar=False)
        dvae = [d for t, d in eventos if t == "bloque" and d["id"] == "dvae"][-1]
        self.assertEqual(dvae["estado"], "omitido")
        self.assertIsNone(corrida.sintesis.mel_dvae)
        self.assertIsNone(corrida.causal)
        self.assertNotIn("causal", [t for t, _ in eventos])

    def test_cancelacion(self):
        detener = threading.Event()
        eventos = []

        def emitir(tipo, datos):
            eventos.append(tipo)
            if tipo == "paso" and datos["paso"] == 3:
                detener.set()

        with self.assertRaises(CorridaCancelada):
            ejecutar_corrida(self.voz, "Hola mundo", ModelosChicos(), emitir, detener=detener,
                             determinista=True, metodo_f0="yin")
        self.assertEqual(eventos.count("paso"), 4)
        self.assertNotIn("etapa4", eventos)


class PruebasWebPipeline(unittest.TestCase):
    """HTTP para lanzar, WebSocket para escuchar, con un servidor real."""

    @classmethod
    def setUpClass(cls):
        cls.carpeta = tempfile.TemporaryDirectory()
        dvae_aleatorio(cls.carpeta.name)
        cls.voz = wav(voz_prueba(3.0))
        cls.app = crear_app()
        modelos = cls.app.extensions["modelos"]
        cls.generador = modelos.obtener("generador", generador_chico)
        modelos.obtener("identidad", ModeloIdentidadChico)
        modelos.obtener("fases", lambda: ReconstructorFases(cls.carpeta.name, ModeloIdentidadChico.normas))
        modelos.obtener("sintetizador", lambda: sintetizador_chico(cls.generador))
        cls.servidor = make_server("127.0.0.1", 0, cls.app, threaded=True)
        cls.puerto = cls.servidor.server_port
        cls.hilo = threading.Thread(target=cls.servidor.serve_forever, daemon=True)
        cls.hilo.start()

    @classmethod
    def tearDownClass(cls):
        cls.servidor.shutdown()
        cls.carpeta.cleanup()

    def setUp(self):
        self.galletas = http.cookiejar.CookieJar()
        self.abridor = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.galletas))
        self.abridor.open(self.url("/")).read()

    def url(self, ruta, esquema="http"):
        return f"{esquema}://127.0.0.1:{self.puerto}{ruta}"

    def lanzar(self, **campos):
        """POST multipart a /api/inferencia; devuelve (estado, json)."""
        campos = {"consentimiento": "si", "texto": "Hola mundo", "determinista": "si",
                  "metodo_f0": "yin", **campos}
        limite = uuid.uuid4().hex
        cuerpo = b""
        for nombre, valor in campos.items():
            cuerpo += (f"--{limite}\r\nContent-Disposition: form-data; name=\"{nombre}\"\r\n\r\n"
                       f"{valor}\r\n").encode()
        if campos.get("con_audio", True):
            cuerpo += (f"--{limite}\r\nContent-Disposition: form-data; name=\"audio\"; "
                       "filename=\"voz.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + self.voz + b"\r\n"
        cuerpo += f"--{limite}--\r\n".encode()
        peticion = urllib.request.Request(self.url("/api/inferencia"), data=cuerpo, headers={
            "Content-Type": f"multipart/form-data; boundary={limite}"})
        try:
            with self.abridor.open(peticion) as respuesta:
                return respuesta.status, json.load(respuesta)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def escuchar(self, ruta, al_evento=None):
        galleta = "; ".join(f"{c.name}={c.value}" for c in self.galletas)
        ws = Client.connect(self.url(ruta, "ws"), headers={"Cookie": galleta})
        eventos = []
        try:
            while True:
                evento = json.loads(ws.receive(timeout=120))
                eventos.append(evento)
                if al_evento:
                    al_evento(ws, evento)
        except ConnectionClosed:
            pass
        return eventos

    def test_corrida_por_http_y_websocket(self):
        estado, datos = self.lanzar()
        self.assertEqual(estado, 202)
        self.assertEqual(datos["ws"], "/ws/corrida/" + datos["id_corrida"])
        eventos = self.escuchar(datos["ws"])
        tipos = [e["tipo"] for e in eventos]
        self.assertEqual(tipos[0], "corrida")
        self.assertEqual(tipos[-1], "terminada")
        self.assertEqual(eventos[-1]["datos"]["motivo"], "completa")
        for tipo in ("etapa1", "etapa2", "inicio", "paso", "fin", "etapa4", "causal"):
            self.assertIn(tipo, tipos)
        self.assertEqual([b["id"] for b in eventos[0]["datos"]["bloques"]], [b[0] for b in BLOQUES])
        tiempos = [e["t"] for e in eventos]
        self.assertEqual(tiempos, sorted(tiempos))
        porTipo = {e["tipo"]: e["datos"] for e in eventos}
        self.assertEqual(porTipo["etapa1"]["id_corrida"], datos["id_corrida"])
        self.assertIn("mel", porTipo["etapa1"]["graficas"])
        self.assertIn("atencion", porTipo["etapa2"]["resultados"][0]["graficas"])
        self.assertTrue(porTipo["etapa4"]["audio"].startswith("data:audio/wav;base64,"))

        with self.abridor.open(self.url(f"/api/inferencia/{datos['id_corrida']}/audio")) as respuesta:
            self.assertEqual(respuesta.headers["Content-Type"], "audio/wav")
            self.assertTrue(respuesta.read().startswith(b"RIFF"))

        # La corrida dejó todo en la memoria de la sesión: las rutas sueltas siguen sirviendo.
        peticion = urllib.request.Request(self.url("/api/explicar/atencion"), headers={
            "Content-Type": "application/json"}, data=json.dumps({
                "consentimiento": "si", "id_corrida": datos["id_corrida"], "capa": 2, "cabeza": 1}).encode())
        with self.abridor.open(peticion) as respuesta:
            self.assertEqual(json.load(respuesta)["filas"], tipos.count("paso"))

        # Quien se conecta después recibe la corrida entera desde el principio.
        self.assertEqual([e["tipo"] for e in self.escuchar(datos["ws"])], tipos)

    def test_detener_por_el_canal(self):
        estado, datos = self.lanzar()
        self.assertEqual(estado, 202)

        def al_evento(ws, evento):
            if evento["tipo"] == "paso" and evento["datos"]["paso"] == 2:
                ws.send(json.dumps({"accion": "detener"}))

        eventos = self.escuchar(datos["ws"], al_evento)
        self.assertEqual(eventos[-1]["tipo"], "terminada")
        self.assertEqual(eventos[-1]["datos"]["motivo"], "cancelada")
        self.assertNotIn("etapa4", [e["tipo"] for e in eventos])

    def test_grabaciones_de_respaldo(self):
        with tempfile.TemporaryDirectory() as carpeta:
            os.environ["CLONVOZ_GRABACIONES"] = carpeta
            try:
                (Path(carpeta) / "respaldo.json").write_text('{"formato": "x"}', encoding="utf-8")
                with self.abridor.open(self.url("/api/grabaciones")) as respuesta:
                    lista = json.load(respuesta)["grabaciones"]
                self.assertEqual([g["nombre"] for g in lista], ["respaldo.json"])
                with self.abridor.open(self.url("/api/grabaciones/respaldo.json")) as respuesta:
                    self.assertEqual(json.load(respuesta), {"formato": "x"})
                for nombre in ("no-existe.json", "..%2Fapp.py", "%2E%2E%5Capp.py"):
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        self.abridor.open(self.url("/api/grabaciones/" + nombre))
                    self.assertEqual(error.exception.code, 404)
            finally:
                del os.environ["CLONVOZ_GRABACIONES"]

    def test_rechazos(self):
        self.assertEqual(self.lanzar(consentimiento="no")[0], 400)
        self.assertEqual(self.lanzar(texto="")[0], 400)
        self.assertEqual(self.lanzar(texto="x" * 240)[0], 400)
        self.assertEqual(self.lanzar(con_audio="")[0], 400)
        eventos = self.escuchar("/ws/corrida/no-existe")
        self.assertEqual(eventos[0]["tipo"], "error")
        try:
            self.abridor.open(self.url("/api/inferencia/no-existe/audio"))
            self.fail("debía responder 404")
        except urllib.error.HTTPError as error:
            self.assertEqual(error.code, 404)

    def test_audio_invalido_llega_como_evento_de_error(self):
        original, type(self).voz = self.voz, b"esto no es un wav"
        try:
            estado, datos = self.lanzar()
        finally:
            type(self).voz = original
        self.assertEqual(estado, 202)
        eventos = self.escuchar(datos["ws"])
        self.assertEqual(eventos[-1]["tipo"], "error")
        self.assertTrue(eventos[-1]["datos"]["errores"])


if __name__ == "__main__":
    unittest.main()
