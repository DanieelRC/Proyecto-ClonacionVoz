"""Pruebas de contratos y regresión; no presentan pesos aleatorios como reales."""

import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from clonvoz.identidad import CodificadorIdentidad, ModeloNoDisponible
from clonvoz import caracteristicas
from clonvoz.analisis import analizar
from clonvoz.web.memoria import MemoriaMel
from clonvoz.web.servidor import crear_app


def wav_prueba(segundos=3):
    """Crea un WAV artificial en memoria, con duración controlada y señal audible.

    El seno de 170 Hz permite probar formato/duración sin usar grabaciones de
    personas. BytesIO evita escribir archivos para cada prueba.
    """
    tiempo = np.arange(int(22050 * segundos)) / 22050
    muestras = (0.2 * np.sin(2 * np.pi * 170 * tiempo)).astype(np.float32)
    archivo = io.BytesIO()
    sf.write(archivo, muestras, 22050, format="WAV")
    return archivo.getvalue()


class ModeloPrueba:
    """Sustituye solo la red para aislar el contrato de paso de datos."""
    dispositivo = "cpu"

    def __init__(self):
        self.recibidos = []

    def procesar_mel(self, mel):
        self.recibidos.append(mel)
        return (np.ones((32, 1024), dtype=np.float32),
                np.full((32, mel.shape[1]), 0.5 / mel.shape[1], dtype=np.float32),
                np.full((32, 32), 0.5 / 32, dtype=np.float32))


class PruebasIdentidad(unittest.TestCase):
    """Verifica que el mel fluya realmente de etapa 1 a etapa 2 sin duplicar trabajo."""
    def preparar(self, cliente, nombre="prueba.wav"):
        respuesta = cliente.post("/api/analizar", data={
            "audio": (io.BytesIO(wav_prueba()), nombre), "metodo_f0": "yin", "consentimiento": "si"})
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(len(respuesta.json["graficas"]), 4)
        return respuesta.json["id_corrida"]

    def test_pesos_faltantes_no_producen_resultado(self):
        with tempfile.TemporaryDirectory() as carpeta:
            with self.assertRaises(ModeloNoDisponible):
                CodificadorIdentidad(carpeta)

    def test_etapa2_exige_resultado_de_etapa1(self):
        cliente = crear_app().test_client()
        with patch("clonvoz.web.identidad.CodificadorIdentidad") as constructor:
            self.assertEqual(cliente.post("/api/identidad").status_code, 400)
            self.assertEqual(cliente.post("/api/identidad", data={"consentimiento": "si",
                "audio": (io.BytesIO(wav_prueba()), "prueba.wav")}).status_code, 400)
            self.assertEqual(cliente.post("/api/identidad", json={"consentimiento": "si"}).status_code, 400)
            self.assertEqual(cliente.post("/api/identidad", json={"consentimiento": "si", "id_corrida": "inexistente"}).status_code, 410)
            constructor.assert_not_called()

    def test_flujo_reutiliza_exactamente_el_mel_y_el_id(self):
        app = crear_app()
        cliente = app.test_client()
        modelo = ModeloPrueba()
        with patch("clonvoz.caracteristicas.mel_espectrograma", wraps=caracteristicas.mel_espectrograma) as calculo:
            identificador = self.preparar(cliente)
            self.assertEqual(calculo.call_count, 1)
            with cliente.session_transaction() as sesion:
                entrada = app.extensions["memoria_mel"].obtener(sesion["propietario"], identificador)
            with patch("clonvoz.web.identidad.CodificadorIdentidad", return_value=modelo):
                respuesta = cliente.post("/api/identidad", json={"consentimiento": "si", "id_corrida": identificador})
            self.assertEqual(respuesta.status_code, 200)
            self.assertEqual(calculo.call_count, 1, "Etapa 2 no debe recalcular el mel")
            self.assertIs(modelo.recibidos[0], entrada.mel)
            self.assertEqual(respuesta.json["resultados"][0]["id_corrida"], identificador)
            self.assertEqual(respuesta.json["resultados"][0]["formas"]["atencion"], [32, entrada.mel.shape[1]])

    def test_comparacion_tambien_usa_resultados_previos(self):
        cliente = crear_app().test_client()
        primero = self.preparar(cliente, "primero.wav")
        segundo = self.preparar(cliente, "segundo.wav")
        modelo = ModeloPrueba()
        with patch("clonvoz.web.identidad.CodificadorIdentidad", return_value=modelo), patch(
                "clonvoz.caracteristicas.mel_espectrograma", side_effect=AssertionError("Recalculo")):
            respuesta = cliente.post("/api/identidad", json={"consentimiento": "si", "id_corrida": segundo,
                                                              "comparacion_id": primero})
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual([r["archivo"] for r in respuesta.json["resultados"]], ["segundo.wav", "primero.wav"])
        self.assertEqual(len(modelo.recibidos), 2)

    def test_no_se_pueden_usar_o_borrar_resultados_de_otra_sesion(self):
        app = crear_app()
        dueño, otro = app.test_client(), app.test_client()
        identificador = self.preparar(dueño)
        otro.post("/api/descartar", json={"id_corrida": identificador})
        with patch("clonvoz.web.identidad.CodificadorIdentidad") as constructor:
            self.assertEqual(otro.post("/api/identidad", json={"consentimiento": "si", "id_corrida": identificador}).status_code, 410)
            constructor.assert_not_called()
        with dueño.session_transaction() as sesion:
            self.assertIsNotNone(app.extensions["memoria_mel"].obtener(sesion["propietario"], identificador))
        dueño.post("/api/descartar", json={"id_corrida": identificador})
        self.assertEqual(dueño.post("/api/identidad", json={"consentimiento": "si", "id_corrida": identificador}).status_code, 410)

    def test_memoria_caduca_y_limita_su_capacidad(self):
        etapa1 = analizar(wav_prueba(), metodo_f0="yin")
        memoria = MemoriaMel(caducidad_s=0.05, capacidad=1)
        memoria.guardar("sesion", "primero", etapa1)
        identificador = etapa1.id_corrida
        etapa1.id_corrida = "segundo"
        memoria.guardar("sesion", "segundo", etapa1)
        with self.assertRaises(KeyError):
            memoria.obtener("sesion", identificador)
        threading.Event().wait(0.2)
        self.assertEqual(len(memoria.entradas), 0)

    def test_etapa1_sigue_funcionando_y_modelo_ausente_da_503(self):
        cliente = crear_app().test_client()
        self.assertEqual(cliente.get("/").status_code, 200)
        identificador = self.preparar(cliente)
        with tempfile.TemporaryDirectory() as carpeta, patch.dict("os.environ", {"CLONVOZ_MODELO": carpeta}):
            respuesta = cliente.post("/api/identidad", json={"consentimiento": "si", "id_corrida": identificador})
        self.assertEqual(respuesta.status_code, 503)


if __name__ == "__main__":
    unittest.main()
