"""Pruebas de la etapa 4 con el GPT y el HiFi-GAN oficiales en tamaño reducido.

Comprueban que se llama el camino real de `Xtts.inference` (latentes -> vocoder)
y la maquinaria web; la calidad la comprueba scripts/verificar_etapa4.py.
"""

import io
import tempfile
import unittest
import warnings

import numpy as np
import soundfile as sf

try:
    import torch
    import TTS  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Las pruebas de la etapa 4 necesitan PyTorch y coqui-tts.")

from clonvoz import config
from clonvoz.fases import ReconstructorFases
from clonvoz.generacion import GeneradorOcupado
from clonvoz.graficas_sintesis import renderizar_sintesis
from clonvoz.sintesis import Sintetizador
from clonvoz.web.servidor import crear_app
from test_generacion import (DIMENSION, ModeloIdentidadChico, dvae_aleatorio, generador_chico,
                             leer_sse, vectores_prueba, voz_prueba, wav)

warnings.filterwarnings("ignore")


def decodificador_chico():
    from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
    torch.manual_seed(0)
    return HifiDecoder(decoder_input_dim=DIMENSION, upsample_initial_channel_decoder=32)


def sintetizador_chico(generador):
    return Sintetizador.desde_modulos(decodificador_chico(), generador)


class PruebasSintetizador(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generador = generador_chico()
        cls.sintetizador = sintetizador_chico(cls.generador)
        cls.vectores = vectores_prueba()
        cls.resultado = cls.generador.generar(cls.vectores, "Hola mundo", determinista=True)
        cls.voz = voz_prueba(3.0)
        cls.embedding = cls.sintetizador.embedding_hablante(cls.voz)

    def test_embedding_normalizado(self):
        self.assertEqual(self.embedding.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(self.embedding)), 1.0, places=4)

    def test_mismo_camino_que_xtts_inference(self):
        """Las mismas llamadas que hace `Xtts.inference` después de `gpt.generate`."""
        gpt, tokens, ids = self.generador.gpt, self.resultado.tokens, self.resultado.texto_ids
        with torch.inference_mode():
            texto = torch.IntTensor(ids)[None]
            codigos = torch.from_numpy(tokens)[None]
            largo = torch.tensor([codigos.shape[-1] * gpt.code_stride_len])
            latentes = gpt(texto, torch.tensor([texto.shape[-1]]), codigos, largo,
                           cond_latents=torch.from_numpy(self.vectores)[None],
                           return_attentions=False, return_latent=True)
            g = torch.from_numpy(self.embedding)[None, :, None]
            onda = self.sintetizador.decodificador(latentes, g=g).squeeze().numpy()

        sintesis = self.sintetizador.sintetizar(self.vectores, ids, tokens, self.embedding)
        np.testing.assert_allclose(sintesis.latentes, latentes[0].numpy(), atol=1e-6)
        np.testing.assert_allclose(sintesis.onda, onda, atol=1e-6)
        self.assertEqual(sintesis.sr, 24000)
        self.assertEqual(sintesis.latentes.shape[1], DIMENSION)
        # Cada latente son 1024 muestras a 22 050 Hz, llevadas a 24 000 Hz.
        esperado = sintesis.latentes.shape[0] * 1024 * 24000 / 22050
        self.assertLess(abs(len(sintesis.onda) - esperado), 0.02 * esperado + 512)
        self.assertEqual(sintesis.mel_salida.shape[0], 80)
        self.assertIsNone(sintesis.mel_dvae)
        self.assertIsNone(sintesis.distancia_mel)

    def test_el_vector_de_hablante_cambia_la_onda(self):
        latentes = self.sintetizador.latentes(self.vectores, self.resultado.texto_ids, self.resultado.tokens)
        con = self.sintetizador.onda(latentes, self.embedding)
        sin = self.sintetizador.onda(latentes, np.zeros_like(self.embedding))
        self.assertEqual(con.shape, sin.shape)
        self.assertGreater(float(np.abs(con - sin).max()), 0)

    def test_con_dvae_y_figuras(self):
        with tempfile.TemporaryDirectory() as carpeta:
            dvae_aleatorio(carpeta)
            fases = ReconstructorFases(carpeta, ModeloIdentidadChico.normas)
            sintesis = self.sintetizador.sintetizar(self.vectores, self.resultado.texto_ids,
                                                    self.resultado.tokens, self.embedding, fases)
        codigos = int((self.resultado.tokens < 1024).sum())
        self.assertEqual(sintesis.mel_dvae.shape, (80, 4 * codigos))
        self.assertGreater(sintesis.distancia_mel, 0)
        resumen = sintesis.como_dict()
        self.assertEqual(resumen["formas"]["embedding"], [512])
        figuras = renderizar_sintesis(sintesis, np.zeros((80, 120), np.float32), temas=("claro",))
        for nombre in ("mels", "onda"):
            self.assertTrue(figuras[nombre]["claro"].startswith(b"\x89PNG"))

    def test_rechazos(self):
        with self.assertRaises(ValueError):
            self.sintetizador.latentes(self.vectores, self.resultado.texto_ids, [])
        with self.generador.cerrojo, self.assertRaises(GeneradorOcupado):
            self.sintetizador.latentes(self.vectores, self.resultado.texto_ids, self.resultado.tokens)


class PruebasWebSintesis(unittest.TestCase):
    """La etapa 4 usa lo que guardaron las etapas 2 y 3 y pide el WAV una sola vez."""

    @classmethod
    def setUpClass(cls):
        cls.carpeta = tempfile.TemporaryDirectory()
        dvae_aleatorio(cls.carpeta.name)
        cls.voz = wav(voz_prueba(3.0))

    @classmethod
    def tearDownClass(cls):
        cls.carpeta.cleanup()

    def setUp(self):
        self.app = crear_app()
        modelos = self.app.extensions["modelos"]
        generador = modelos.obtener("generador", generador_chico)
        modelos.obtener("identidad", ModeloIdentidadChico)
        modelos.obtener("fases", lambda: ReconstructorFases(self.carpeta.name, ModeloIdentidadChico.normas))
        modelos.obtener("sintetizador", lambda: sintetizador_chico(generador))
        self.cliente = self.app.test_client()
        respuesta = self.cliente.post("/api/analizar", data={
            "audio": (io.BytesIO(self.voz), "voz.wav"), "metodo_f0": "yin", "consentimiento": "si"})
        self.id = respuesta.json["id_corrida"]

    def sintetizar(self, audio=None, **extra):
        datos = {"consentimiento": "si", "id_corrida": self.id, **extra}
        if audio is not None:
            datos["audio"] = (io.BytesIO(audio), "voz.wav")
        return self.cliente.post("/api/sintetizar", data=datos, content_type="multipart/form-data")

    def contraste(self):
        return self.cliente.post("/api/sintetizar/contraste",
                                 json={"consentimiento": "si", "id_corrida": self.id})

    def test_flujo_completo(self):
        self.assertEqual(self.sintetizar(self.voz).status_code, 409)  # falta la etapa 3
        self.cliente.post("/api/identidad", json={"consentimiento": "si", "id_corrida": self.id})
        fin = leer_sse(self.cliente.post("/api/generar", json={
            "consentimiento": "si", "id_corrida": self.id, "texto": "Hola mundo",
            "determinista": True}).get_data(as_text=True))[-1][1]

        self.assertEqual(self.contraste().status_code, 409)
        self.assertEqual(self.sintetizar().status_code, 400)  # falta el WAV
        self.assertEqual(self.sintetizar(wav(voz_prueba(4.0))).status_code, 409)
        self.assertEqual(self.sintetizar(self.voz, consentimiento="no").status_code, 400)
        self.assertEqual(self.sintetizar(self.voz, id_corrida="otro").status_code, 410)

        respuesta = self.sintetizar(self.voz)
        self.assertEqual(respuesta.status_code, 200)
        datos = respuesta.json
        self.assertEqual(datos["sr"], 24000)
        self.assertEqual(datos["texto"], "Hola mundo")
        self.assertEqual(datos["formas"]["mel_dvae"], [80, 4 * sum(t < 1024 for t in fin["tokens"])])
        self.assertTrue(datos["audio"].startswith("data:audio/wav;base64,"))
        for figura in ("mels", "onda"):
            self.assertTrue(datos["graficas"][figura]["oscuro"].startswith("data:image/png;base64,"))

        # El vector de hablante quedó guardado: ya no hace falta reenviar el WAV.
        self.assertEqual(self.sintetizar().status_code, 200)
        respuesta = self.contraste()
        self.assertEqual(respuesta.status_code, 200)
        info = sf.info(io.BytesIO(respuesta.data))
        self.assertEqual(info.samplerate, 24000)
        self.assertEqual(info.frames, datos["muestras"])

    def test_explicacion_por_la_web(self):
        cuerpo = {"consentimiento": "si", "id_corrida": self.id}
        self.assertEqual(self.cliente.post("/api/explicar/causal", json=cuerpo).status_code, 409)
        self.cliente.post("/api/identidad", json=cuerpo)
        eventos = leer_sse(self.cliente.post("/api/generar", json={
            **cuerpo, "texto": "Hola mundo", "determinista": True}).get_data(as_text=True))
        inicio, pasos = eventos[0][1], [d for t, d in eventos if t == "paso"]
        self.assertEqual((inicio["total_capas"], inicio["cabezas"]), (4, 4))
        self.assertTrue(all("zonas" in p for p in pasos))

        respuesta = self.cliente.post("/api/explicar/atencion", json={**cuerpo, "capa": 3, "cabeza": 2})
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual((respuesta.json["capa"], respuesta.json["cabeza"]), (3, 2))
        self.assertEqual(respuesta.json["filas"], len(pasos))
        self.assertEqual(respuesta.json["columnas"], inicio["prefijo"] + len(pasos) - 1)
        self.assertIsNone(self.cliente.post("/api/explicar/atencion",
                                            json={**cuerpo, "capa": 1}).json["cabeza"])
        self.assertEqual(self.cliente.post("/api/explicar/atencion", json={**cuerpo, "capa": 9}).status_code, 400)
        self.assertEqual(self.cliente.post("/api/explicar/atencion", json=cuerpo).status_code, 400)

        causal = self.cliente.post("/api/explicar/causal", json=cuerpo)
        self.assertEqual(causal.status_code, 200)
        self.assertEqual(causal.json["pasos"], len(pasos))
        cabezas = self.cliente.post("/api/explicar/causal", json={**cuerpo, "capa": 2})
        self.assertEqual(len(cabezas.json["cabezas"]), 4)
        self.assertEqual(self.cliente.post("/api/explicar/causal",
                                           json={**cuerpo, "consentimiento": "no"}).status_code, 400)


class PruebasWav(unittest.TestCase):
    def test_wav_a_24k(self):
        from clonvoz.fases import a_wav
        self.assertEqual(sf.info(io.BytesIO(a_wav(np.ones(10), sr=24000))).samplerate, 24000)
        self.assertEqual(sf.info(io.BytesIO(a_wav(np.ones(10)))).samplerate, config.SR)


if __name__ == "__main__":
    unittest.main()
