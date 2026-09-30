"""Pruebas de la etapa 3 con el GPT oficial en tamaño reducido y pesos aleatorios.

Comprueban la maquinaria (tokens, atención, cancelación, SSE y fases), no la
calidad: eso lo hace scripts/verificar_etapa3.py con los pesos reales.
"""

import base64
import io
import json
import tempfile
import threading
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

try:
    import torch
    import TTS  # noqa: F401
except ImportError:  # entorno de la etapa 1, sin PyTorch
    raise unittest.SkipTest("Las pruebas de la etapa 3 necesitan PyTorch y coqui-tts.")

from clonvoz import caracteristicas, config
from clonvoz.fases import (MUESTRAS_POR_CODIGO, ReconstructorFases, a_wav,
                           audio_desde_mel_referencia)
from clonvoz.generacion import GeneradorOcupado, GeneradorTokens, evento_paso, fila_cuantizada
from clonvoz.identidad import ModeloNoDisponible
from clonvoz.web.servidor import crear_app

warnings.filterwarnings("ignore")
DIMENSION = 64


class TokenizadorPrueba:
    """Sustituye al BPE: ids deterministas a partir de los caracteres."""
    char_limits = {"es": 239}
    tokenizer = None

    def encode(self, texto, lang):
        return [1 + (ord(c) % 90) for c in f"[{lang}]{texto}"]

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


def gpt_chico():
    from TTS.tts.layers.xtts.gpt import GPT
    torch.manual_seed(0)
    return GPT(layers=4, model_dim=DIMENSION, heads=4, max_text_tokens=60, max_mel_tokens=40,
               max_prompt_tokens=10, number_text_tokens=100, start_text_token=98, stop_text_token=0,
               num_audio_tokens=66, start_audio_token=64, stop_audio_token=65,
               use_perceiver_resampler=True, code_stride_len=1024)


def generador_chico():
    return GeneradorTokens.desde_modulos(gpt_chico(), TokenizadorPrueba())


def vectores_prueba():
    return np.random.default_rng(1).standard_normal((32, DIMENSION)).astype(np.float32)


def voz_prueba(segundos=3.0):
    """Tono con armónicos y vibrato: tiene la estructura que el mel debe conservar."""
    t = np.arange(int(config.SR * segundos)) / config.SR
    f0 = 150 + 10 * np.sin(2 * np.pi * 4 * t)
    fase = 2 * np.pi * np.cumsum(f0) / config.SR
    senal = sum(np.sin(k * fase) / k for k in range(1, 8))
    return (0.2 * senal / np.abs(senal).max()).astype(np.float32)


def wav(muestras):
    archivo = io.BytesIO()
    sf.write(archivo, muestras, config.SR, format="WAV")
    return archivo.getvalue()


def dvae_aleatorio(carpeta):
    from TTS.tts.layers.xtts.dvae import DiscreteVAE
    torch.manual_seed(0)
    dvae = DiscreteVAE(channels=80, normalization=None, positional_dims=1, num_tokens=1024,
                       codebook_dim=512, hidden_dim=512, num_resnet_blocks=3, kernel_size=3,
                       num_layers=2, use_transposed_convs=False)
    torch.save(dvae.state_dict(), Path(carpeta) / "dvae.pth")
    return dvae


def leer_sse(texto):
    eventos = []
    for bloque in texto.strip().split("\n\n"):
        tipo, datos = bloque.split("\n", 1)
        eventos.append((tipo.removeprefix("event: "), json.loads(datos.removeprefix("data: "))))
    return eventos


class PruebasGenerador(unittest.TestCase):
    """El GPT instrumentado debe dar exactamente lo mismo que `GPT.generate` oficial."""

    @classmethod
    def setUpClass(cls):
        cls.generador = generador_chico()
        cls.vectores = vectores_prueba()

    def oficial(self, texto, determinista, semilla):
        ids, _ = self.generador.tokenizar(texto)
        extra = {"do_sample": False} if determinista else dict(
            do_sample=True, temperature=0.75, top_k=50, top_p=0.85)
        torch.manual_seed(semilla)
        with torch.inference_mode():
            return self.generador.gpt.generate(
                torch.from_numpy(self.vectores)[None], torch.IntTensor(ids)[None],
                repetition_penalty=10.0, length_penalty=1.0, num_beams=1, **extra)[0].tolist()

    def test_tokens_identicos_al_generate_oficial(self):
        for determinista in (True, False):
            resultado = self.generador.generar(self.vectores, "Hola mundo",
                                               determinista=determinista, semilla=7)
            self.assertEqual(resultado.tokens.tolist(), self.oficial("Hola mundo", determinista, 7))

    def test_atencion_por_paso_y_ganchos_retirados(self):
        pasos = []
        resultado = self.generador.generar(self.vectores, "Hola mundo", al_paso=pasos.append,
                                           determinista=True)
        self.assertEqual(len(pasos), len(resultado.tokens))
        self.assertEqual(self.generador.capas, (0, 1, 3))
        self.assertEqual(resultado.prefijo, 32 + len(self.generador.tokenizar("Hola mundo")[0]) + 3)
        for paso in pasos:
            self.assertEqual(sorted(paso.atencion), [0, 1, 3])
            for fila in paso.atencion.values():
                self.assertEqual(fila.shape, (resultado.prefijo + paso.paso,))
                self.assertAlmostEqual(float(fila.sum()), 1.0, places=4)
            self.assertTrue(0 <= paso.probabilidad <= 1)
        for capa in self.generador.gpt.gpt.h:
            self.assertFalse(capa.attn._forward_hooks)

    def test_cancelacion(self):
        detener = threading.Event()

        def al_paso(paso):
            if paso.paso == 4:
                detener.set()

        resultado = self.generador.generar(self.vectores, "Hola mundo", al_paso=al_paso,
                                           detener=detener, determinista=True)
        self.assertEqual(resultado.motivo, "cancelada")
        self.assertEqual(len(resultado.tokens), 5)

    def test_una_generacion_a_la_vez(self):
        with self.generador.cerrojo:
            with self.assertRaises(GeneradorOcupado):
                self.generador.generar(self.vectores, "Hola")

    def test_entradas_invalidas(self):
        for texto in ("", "   ", "x" * 240):
            with self.assertRaises(ValueError):
                self.generador.tokenizar(texto)
        for vectores in (np.ones((32, 1024), np.float32), np.full((32, DIMENSION), np.nan, np.float32)):
            with self.assertRaises(ValueError):
                self.generador.generar(vectores, "Hola")

    def test_evento_paso_es_json_valido_y_recuperable(self):
        pasos = []
        self.generador.generar(self.vectores, "Hola", al_paso=pasos.append, determinista=True)
        paso = pasos[3]
        paso.probabilidad = float("nan")
        evento = json.loads(json.dumps(evento_paso(paso), allow_nan=False))
        self.assertIsNone(evento["probabilidad"])
        self.assertEqual(sorted(evento["atencion"]), ["1", "2", "4"])
        fila = paso.atencion[3]
        cuantizada = fila_cuantizada(fila)
        q = np.frombuffer(base64.b64decode(cuantizada["q"]), np.uint8)
        np.testing.assert_allclose(q / 255 * cuantizada["max"], fila, atol=cuantizada["max"] / 510 + 1e-7)


class ModeloIdentidadChico:
    """Etapa 2 sustituta con la dimensión del GPT reducido."""
    dispositivo = "cpu"
    normas = np.full(80, 3.0, np.float32)

    def procesar_mel(self, mel):
        return (np.random.default_rng(2).standard_normal((32, DIMENSION)).astype(np.float32),
                np.full((32, mel.shape[1]), 0.5 / mel.shape[1], np.float32),
                np.full((32, 32), 0.5 / 32, np.float32))


class PruebasWebGeneracion(unittest.TestCase):
    """La etapa 3 usa la identidad guardada por la etapa 2 y transmite por SSE."""

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
        self.generador = modelos.obtener("generador", generador_chico)
        modelos.obtener("identidad", ModeloIdentidadChico)
        modelos.obtener("fases", lambda: ReconstructorFases(self.carpeta.name, ModeloIdentidadChico.normas))
        self.cliente = self.app.test_client()
        respuesta = self.cliente.post("/api/analizar", data={
            "audio": (io.BytesIO(self.voz), "voz.wav"), "metodo_f0": "yin", "consentimiento": "si"})
        self.assertEqual(respuesta.status_code, 200)
        self.id = respuesta.json["id_corrida"]

    def generar(self, **cuerpo):
        return self.cliente.post("/api/generar", json={
            "consentimiento": "si", "id_corrida": self.id, "texto": "Hola mundo", **cuerpo})

    def fase(self, fase, audio=None):
        datos = {"consentimiento": "si", "fase": fase, "id_corrida": self.id}
        if audio is not None:
            datos["audio"] = (io.BytesIO(audio), "voz.wav")
        return self.cliente.post("/api/fases/audio", data=datos, content_type="multipart/form-data")

    def test_etapa3_exige_etapa2_y_transmite_en_orden(self):
        self.assertEqual(self.generar().status_code, 409)
        self.assertEqual(self.cliente.post("/api/identidad", json={
            "consentimiento": "si", "id_corrida": self.id}).status_code, 200)

        respuesta = self.generar(determinista=True)
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(respuesta.mimetype, "text/event-stream")
        eventos = leer_sse(respuesta.get_data(as_text=True))
        tipos = [tipo for tipo, _ in eventos]
        self.assertEqual(tipos[0], "inicio")
        self.assertEqual(tipos[-1], "fin")
        self.assertEqual(set(tipos[1:-1]), {"paso"})
        inicio, fin = eventos[0][1], eventos[-1][1]
        self.assertEqual(inicio["capas"], [1, 2, 4])
        pasos = [datos for tipo, datos in eventos if tipo == "paso"]
        self.assertEqual([p["paso"] for p in pasos], list(range(len(pasos))))
        self.assertEqual(fin["tokens"], [p["token"] for p in pasos])
        largo = len(base64.b64decode(pasos[-1]["atencion"]["4"]["q"]))
        self.assertEqual(largo, inicio["prefijo"] + len(pasos) - 1)

        with self.cliente.session_transaction() as sesion:
            entrada = self.app.extensions["memoria_mel"].obtener(sesion["propietario"], self.id)
        self.assertEqual(entrada.tokens_audio.tolist(), fin["tokens"])

    def test_rechazos(self):
        self.cliente.post("/api/identidad", json={"consentimiento": "si", "id_corrida": self.id})
        self.assertEqual(self.generar(consentimiento="no").status_code, 400)
        self.assertEqual(self.generar(id_corrida="otro").status_code, 410)
        self.assertEqual(self.generar(texto="").status_code, 400)
        with self.generador.cerrojo:
            self.assertEqual(self.generar().status_code, 409)

    def test_fases_audibles(self):
        respuesta = self.fase("mel")
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(respuesta.mimetype, "audio/wav")
        self.assertEqual(self.fase("tokens_texto").status_code, 409)
        self.assertEqual(self.fase("tokens_referencia").status_code, 400)
        self.assertEqual(self.fase("tokens_referencia", wav(voz_prueba(4.0))).status_code, 409)
        self.assertEqual(self.fase("tokens_referencia", self.voz).status_code, 200)
        self.assertEqual(self.fase("otra").status_code, 400)

        self.cliente.post("/api/identidad", json={"consentimiento": "si", "id_corrida": self.id})
        fin = leer_sse(self.generar(determinista=True).get_data(as_text=True))[-1][1]
        respuesta = self.fase("tokens_texto")
        self.assertEqual(respuesta.status_code, 200)
        codigos = [t for t in fin["tokens"] if t < 1024]
        info = sf.info(io.BytesIO(respuesta.data))
        self.assertEqual(info.frames, len(codigos) * MUESTRAS_POR_CODIGO - config.HOP_LENGTH)


class PruebasFases(unittest.TestCase):
    """Griffin-Lim y el DVAE: parámetros de inversión y formas."""

    def test_fase2_invierte_el_mismo_mel(self):
        # Se mide donde hay energía: en el piso del log-mel Griffin-Lim solo deja ruido.
        # Con estos parámetros el error es ~0.23; con la escala mel slaney sube a ~1.5
        # y sin la normalización a ~3.6.
        voz = voz_prueba(3.0)
        mel = caracteristicas.mel_espectrograma(voz)
        otra_vez = caracteristicas.mel_espectrograma(audio_desde_mel_referencia(mel))
        cuadros = min(mel.shape[1], otra_vez.shape[1])
        original, reconstruido = mel[:, :cuadros], otra_vez[:, :cuadros]
        con_energia = original > original.max() - 6
        self.assertLess(np.abs(original - reconstruido)[con_energia].mean(), 0.5)

    def test_fases_3_y_4_formas(self):
        with tempfile.TemporaryDirectory() as carpeta:
            dvae_aleatorio(carpeta)
            fases = ReconstructorFases(carpeta, np.full(80, 3.0, np.float32))
            voz = voz_prueba(2.0)
            codigos = fases.tokens_de_referencia(voz)
            cuadros = len(voz) // config.HOP_LENGTH + 1
            self.assertEqual(len(codigos), -(-cuadros // 4))  # 4 cuadros de mel por código
            self.assertTrue(((codigos >= 0) & (codigos < 1024)).all())
            tokens = np.array([5, 17, 900, 1025])
            self.assertEqual(fases.mel_desde_tokens(tokens).shape, (80, 12))
            self.assertEqual(len(fases.audio_desde_tokens(tokens)), 3 * MUESTRAS_POR_CODIGO - config.HOP_LENGTH)
            with self.assertRaises(ValueError):
                fases.mel_desde_tokens([1024, 1025])

    def test_dvae_incompleto_se_rechaza(self):
        with tempfile.TemporaryDirectory() as carpeta:
            estado = dvae_aleatorio(carpeta).state_dict()
            torch.save({k: v for k, v in estado.items() if not k.startswith("decoder.")},
                       Path(carpeta) / "dvae.pth")
            with self.assertRaises(ModeloNoDisponible):
                ReconstructorFases(carpeta, np.full(80, 3.0, np.float32))
        with tempfile.TemporaryDirectory() as carpeta, self.assertRaises(ModeloNoDisponible):
            ReconstructorFases(carpeta, np.full(80, 3.0, np.float32))

    def test_wav_normalizado(self):
        datos, sr = sf.read(io.BytesIO(a_wav(np.array([0.0, 2.0, -4.0]))))
        self.assertEqual(sr, config.SR)
        self.assertAlmostEqual(float(np.abs(datos).max()), 0.9, places=3)


if __name__ == "__main__":
    unittest.main()
