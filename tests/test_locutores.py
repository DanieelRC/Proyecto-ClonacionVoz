"""Etapa 7: reconocimiento del locutor con vectores sintéticos (sin pesos de XTTS-v2)."""

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from clonvoz import config
from clonvoz.identidad import ModeloNoDisponible
from clonvoz.locutores import (ARCHIVO_EMBEDDINGS, DESCONOCIDO, Conjunto, Muestra, asignar_pliegues,
                               cargar_reconocedor, entrenar, extraer_conjunto, guardar, leer_audio)


def conjunto_sintetico(locutores=6, audios=5, dimension=1024, ruido=0.6, semilla=0):
    """Cada locutor = un desplazamiento propio sobre 32 consultas comunes a todos, más ruido."""
    rng = np.random.default_rng(semilla)
    consultas = rng.standard_normal((32, dimension)) * 3
    muestras = []
    for i in range(locutores):
        centro = rng.standard_normal(dimension)
        for j in range(audios):
            vectores = consultas + centro + ruido * rng.standard_normal((32, dimension))
            muestras.append(Muestra(f"locutor_{i:02d}", f"locutor_{i:02d}/audio_{j}.wav", 6.0,
                                    vectores.astype(np.float32)))
    return Conjunto(muestras, carpeta="sintetico"), consultas


def reconocedor_chico(dimension=1024):
    """Un reconocedor entrenado de verdad, para que el pipeline lo use en las pruebas."""
    conjunto, _ = conjunto_sintetico(locutores=3, audios=4, dimension=dimension)
    return entrenar(conjunto, componentes=2, dimensiones=8).reconocedor


def wav(muestras, sr=config.SR):
    archivo = io.BytesIO()
    sf.write(archivo, muestras, sr, format="WAV")
    return archivo.getvalue()


def tono(segundos, sr=config.SR, frecuencia=170.0):
    tiempo = np.arange(int(sr * segundos)) / sr
    return (0.2 * np.sin(2 * np.pi * frecuencia * tiempo)).astype(np.float32)


class CodificadorFalso:
    """Sustituye al Perceiver: un vector distinto según la frecuencia del tono."""
    dispositivo = "cpu"

    def __init__(self):
        self.llamadas = 0

    def procesar_mel(self, mel):
        self.llamadas += 1
        semilla = int(np.argmax(mel.mean(axis=1)))
        vectores = np.random.default_rng(semilla).standard_normal((32, 1024)).astype(np.float32)
        return vectores, None, None


class PruebasEntrenamiento(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conjunto, cls.consultas = conjunto_sintetico()
        cls.resultado = entrenar(cls.conjunto)
        cls.reporte = cls.resultado.reporte

    def test_pliegues_un_audio_de_prueba_por_locutor(self):
        asignacion = asignar_pliegues(self.conjunto, 5)
        for pliegue in range(5):
            locutores = [m.locutor for m, p in zip(self.conjunto.muestras, asignacion) if p == pliegue]
            self.assertEqual(sorted(locutores), self.conjunto.locutores)

    def test_ambos_modelos_reconocen_y_la_matriz_cuenta_cada_audio(self):
        for clave in ("knn", "gmm"):
            modelo = self.reporte["modelos"][clave]
            self.assertGreaterEqual(modelo["exactitud"], 0.9, clave)
            matriz = np.array(modelo["matriz_confusion"])
            self.assertEqual(matriz.shape, (6, 6))
            self.assertTrue((matriz.sum(axis=1) == 5).all())
            self.assertAlmostEqual(modelo["exactitud"], np.trace(matriz) / matriz.sum())
            self.assertEqual(len(modelo["exactitud_pliegues"]), 5)
            self.assertIsNotNone(modelo["umbral"])

    def test_busqueda_cubre_ambas_distancias(self):
        combinaciones = {(f["k"], f["metrica"]) for f in self.reporte["busqueda_knn"]}
        self.assertEqual(combinaciones, {(k, m) for k in config.LOCUTOR_K for m in config.LOCUTOR_METRICAS})
        self.assertIn(self.reporte["elegido"], ("knn", "gmm"))
        self.assertIn(self.reporte["modelos"][self.reporte["elegido"]]["descripcion"],
                      self.reporte["motivo_eleccion"])

    def test_identifica_y_rechaza_desconocidos(self):
        reconocedor = self.resultado.reconocedor
        conocido = self.conjunto.muestras[7]
        resultado = reconocedor.identificar(conocido.vectores)
        self.assertEqual(resultado.locutor, conocido.locutor)
        self.assertEqual(resultado.modelo, self.reporte["elegido"])
        self.assertEqual(len(resultado.candidatos), 3)
        self.assertEqual(resultado.otro["mas_parecido"], conocido.locutor)

        rng = np.random.default_rng(99)
        extrano = self.consultas + 4 * rng.standard_normal(1024) + 0.6 * rng.standard_normal((32, 1024))
        resultado = reconocedor.identificar(extrano.astype(np.float32))
        self.assertEqual(resultado.locutor, DESCONOCIDO)
        self.assertTrue(resultado.desconocido)
        json.dumps(resultado.como_dict(), allow_nan=False)

    def test_forma_invalida(self):
        with self.assertRaises(ValueError):
            self.resultado.reconocedor.identificar(np.zeros((32, 512), np.float32))

    def test_guardar_cargar_y_exportar(self):
        with tempfile.TemporaryDirectory() as carpeta:
            escritos = guardar(self.resultado, carpeta)
            nombres = {p.name for p in escritos}
            for nombre in ("matriz_confusion_knn.png", "matriz_confusion_gmm.png", "matriz_confusion_knn.csv",
                           "metricas.csv", "busqueda_knn.csv", "recall_por_locutor.csv", "predicciones.csv",
                           "proyeccion_pca.png", "reporte.json"):
                self.assertIn(nombre, nombres)
            cargado = cargar_reconocedor(carpeta)
            muestra = self.conjunto.muestras[0]
            self.assertEqual(cargado.identificar(muestra.vectores).locutor, muestra.locutor)
            filas = (Path(carpeta) / "reporte" / "metricas.csv").read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(filas), 3)

    def test_sin_modelo(self):
        with tempfile.TemporaryDirectory() as carpeta, self.assertRaises(ModeloNoDisponible):
            cargar_reconocedor(carpeta)


class PruebasAudios(unittest.TestCase):
    def test_wav_estereo_a_44100(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "a.wav"
            estereo = np.stack([tono(2.0, 44100)] * 2, axis=1)
            sf.write(ruta, estereo, 44100)
            muestras = leer_audio(ruta)
            self.assertEqual(muestras.dtype, np.float32)
            self.assertAlmostEqual(len(muestras) / config.SR, 2.0, places=2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "requiere FFmpeg")
    def test_mp3_y_m4a_con_ffmpeg(self):
        with tempfile.TemporaryDirectory() as carpeta:
            origen = Path(carpeta) / "origen.wav"
            sf.write(origen, tono(3.0, 44100), 44100)
            for extension in ("mp3", "m4a"):
                destino = Path(carpeta) / f"voz.{extension}"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(origen), str(destino)], check=True)
                muestras = leer_audio(destino)
                self.assertAlmostEqual(len(muestras) / config.SR, 3.0, delta=0.1, msg=extension)
                self.assertGreater(np.abs(muestras).max(), 0.1)

    def test_extraer_conjunto_rechaza_y_reutiliza_la_cache(self):
        with tempfile.TemporaryDirectory() as carpeta:
            raiz = Path(carpeta) / "datos"
            for indice, locutor in enumerate(("ana", "beto")):
                (raiz / locutor).mkdir(parents=True)
                for j in range(3):
                    sf.write(raiz / locutor / f"{j}.wav", tono(2.0, frecuencia=150 + 400 * indice + 20 * j), config.SR)
            sf.write(raiz / "ana" / "corto.wav", tono(0.4), config.SR)       # menos de 1 s
            (raiz / "ana" / "notas.txt").write_text("no es audio")           # se ignora
            (raiz / "solo").mkdir()
            sf.write(raiz / "solo" / "0.wav", tono(2.0), config.SR)            # un solo audio

            codificador = CodificadorFalso()
            cache = Path(carpeta) / ARCHIVO_EMBEDDINGS
            conjunto = extraer_conjunto(raiz, codificador, cache=cache)
            self.assertEqual(conjunto.locutores, ["ana", "beto"])
            self.assertEqual(len(conjunto.muestras), 6)
            rechazados = {r["archivo"] for r in conjunto.rechazados}
            self.assertEqual(rechazados, {"ana/corto.wav", "solo/0.wav"})
            self.assertEqual(codificador.llamadas, 7)

            otra = extraer_conjunto(raiz, codificador, cache=cache)
            self.assertEqual(codificador.llamadas, 8)  # solo el locutor de un audio se vuelve a calcular
            np.testing.assert_array_equal(otra.muestras[0].vectores, conjunto.muestras[0].vectores)


class PruebasWebLocutores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.carpeta = tempfile.TemporaryDirectory()
        conjunto, _ = conjunto_sintetico(locutores=4, audios=5)
        cls.muestra = conjunto.muestras[3]
        guardar(entrenar(conjunto), cls.carpeta.name)

    @classmethod
    def tearDownClass(cls):
        cls.carpeta.cleanup()

    def cliente(self, carpeta):
        from clonvoz.web.servidor import crear_app
        with patch.dict(os.environ, {"CLONVOZ_LOCUTORES": carpeta}):
            app = crear_app()
        app.testing = True
        return app, app.test_client()

    def test_sin_entrenar(self):
        with tempfile.TemporaryDirectory() as vacia:
            _, cliente = self.cliente(vacia)
            datos = cliente.get("/api/locutores").get_json()
            self.assertFalse(datos["hay_modelo"])

    def test_reporte_y_exportacion(self):
        _, cliente = self.cliente(self.carpeta.name)
        datos = cliente.get("/api/locutores").get_json()
        self.assertTrue(datos["hay_modelo"])
        self.assertEqual(set(datos["graficas"]), {"matriz_knn", "matriz_gmm", "proyeccion"})
        self.assertTrue(datos["graficas"]["matriz_knn"]["oscuro"].startswith("data:image/png"))
        self.assertIn("metricas.csv", datos["exportados"])

        respuesta = cliente.get("/api/locutores/exportar/metricas.csv")
        self.assertEqual(respuesta.status_code, 200)
        self.assertIn("attachment", respuesta.headers["Content-Disposition"])
        self.assertEqual(cliente.get("/api/locutores/exportar/..%2Freconocedor.joblib").status_code, 404)
        self.assertEqual(cliente.get("/api/locutores/exportar/no_existe.csv").status_code, 404)

        comprimido = zipfile.ZipFile(io.BytesIO(cliente.get("/api/locutores/exportar.zip").data))
        self.assertIn("matriz_confusion_gmm.png", comprimido.namelist())

    def test_identificar_la_referencia_de_la_etapa_2(self):
        app, cliente = self.cliente(self.carpeta.name)
        with cliente.session_transaction() as sesion:
            sesion["propietario"] = "persona"
        memoria = app.extensions["memoria_mel"]
        analisis = SimpleNamespace(id_corrida="c1", mel=np.zeros((80, 10), np.float32),
                                   audio=SimpleNamespace(duracion_s=6.0),
                                   validacion=SimpleNamespace(advertencias=[]))
        memoria.guardar("persona", "voz.wav", analisis)

        cuerpo = {"id_corrida": "c1", "consentimiento": "si"}
        self.assertEqual(cliente.post("/api/locutor", json=cuerpo).status_code, 409)  # falta la etapa 2
        memoria.anotar("persona", "c1", identidad=self.muestra.vectores)
        respuesta = cliente.post("/api/locutor", json=cuerpo)
        self.assertEqual(respuesta.status_code, 200)
        datos = respuesta.get_json()
        self.assertEqual(datos["locutor"], self.muestra.locutor)
        self.assertEqual(datos["archivo"], "voz.wav")
        self.assertIn("proyeccion", datos["graficas"])
        self.assertEqual(cliente.post("/api/locutor", json={"id_corrida": "otra", "consentimiento": "si"}).status_code, 410)
        self.assertEqual(cliente.post("/api/locutor", json={"id_corrida": "c1"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
