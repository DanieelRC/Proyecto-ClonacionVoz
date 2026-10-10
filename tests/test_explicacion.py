"""Pruebas de la interpretabilidad ampliada con el GPT oficial reducido.

Comprueban que la pasada con teacher forcing reproduce lo que el modelo vio al
generar y que las intervenciones cambian lo que deben cambiar.
"""

import base64
import json
import unittest

import numpy as np

try:
    import torch  # noqa: F401
    import TTS  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Estas pruebas necesitan PyTorch y coqui-tts.")

from clonvoz.explicacion import Explicador, matriz_cuantizada
from clonvoz.generacion import GeneradorOcupado, evento_paso
from test_generacion import generador_chico, vectores_prueba


class PruebasExplicador(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generador = generador_chico()
        cls.explicador = Explicador(cls.generador)
        cls.vectores = vectores_prueba()
        cls.pasos = []
        cls.resultado = cls.generador.generar(cls.vectores, "Hola mundo", al_paso=cls.pasos.append,
                                              determinista=True)
        cls.args = (cls.vectores, cls.resultado.texto_ids, cls.resultado.tokens)

    def test_zonas_en_vivo_por_cabeza_y_capa(self):
        fin_texto = self.resultado.prefijo - 1
        for paso in self.pasos:
            self.assertEqual(paso.zonas.shape, (4, 4, 3))
            np.testing.assert_allclose(paso.zonas.sum(axis=2), 1, atol=1e-4)
            # El promedio de cabezas de las zonas coincide con la fila promedio ya existente.
            for capa, fila in paso.atencion.items():
                esperado = [fila[:32].sum(), fila[32:fin_texto].sum(), fila[fin_texto:].sum()]
                np.testing.assert_allclose(paso.zonas[capa].mean(axis=0), esperado, atol=1e-5)
        evento = json.loads(json.dumps(evento_paso(self.pasos[2])))
        q = np.frombuffer(base64.b64decode(evento["zonas"]), np.uint8).reshape(4, 4, 3)
        np.testing.assert_allclose(q / 255, self.pasos[2].zonas, atol=1 / 255)

    def test_atencion_completa_reproduce_la_generacion(self):
        for capa in self.generador.capas:
            matriz = self.explicador.atencion(*self.args, capa)
            self.assertEqual(matriz.shape, (len(self.pasos), self.resultado.prefijo + len(self.pasos) - 1))
            for paso in self.pasos:
                largo = self.resultado.prefijo + paso.paso
                np.testing.assert_allclose(matriz[paso.paso, :largo], paso.atencion[capa], atol=1e-5)
                self.assertEqual(float(np.abs(matriz[paso.paso, largo:]).max(initial=0)), 0)
        # Una capa que el mapa en vivo no muestra, cabeza por cabeza.
        cabezas = np.stack([self.explicador.atencion(*self.args, 2, c) for c in range(4)])
        np.testing.assert_allclose(cabezas.mean(axis=0), self.explicador.atencion(*self.args, 2), atol=1e-6)
        for paso in self.pasos:
            fin_texto = self.resultado.prefijo - 1
            np.testing.assert_allclose(cabezas[:, paso.paso, :32].sum(axis=1), paso.zonas[2, :, 0], atol=1e-5)
            np.testing.assert_allclose(cabezas[:, paso.paso, 32:fin_texto].sum(axis=1),
                                       paso.zonas[2, :, 1], atol=1e-5)
        for capa in self.generador.gpt.gpt.h:
            self.assertFalse(capa.attn._forward_hooks)

    def test_log_p_base_predice_los_tokens_deterministas(self):
        def calcular():
            entrada, _, prefijo = self.explicador._preparar(*self.args)
            return self.explicador._logaritmos(entrada, prefijo)
        base = self.explicador._con_cerrojo(calcular)
        self.assertEqual(base.shape[0], len(self.resultado.tokens))
        # El primer token no tiene penalización por repetición: debe ser el más probable.
        self.assertEqual(int(base[0].argmax()), int(self.resultado.tokens[0]))

    def test_efecto_causal(self):
        efecto = self.explicador.efecto_causal(*self.args)
        json.dumps(efecto, allow_nan=False)
        pasos = len(self.resultado.tokens)
        self.assertEqual(efecto["pasos"], pasos)
        self.assertEqual(len(efecto["intervenciones"]), 2 * 4)
        self.assertEqual({i["zona"] for i in efecto["intervenciones"]}, {"voz", "texto"})
        for zona in ("voz", "texto", "audio"):
            self.assertEqual(len(efecto["atencion"][zona]), pasos)
        np.testing.assert_allclose(sum(np.array(efecto["atencion"][z]) for z in ("voz", "texto", "audio")),
                                   1, atol=1e-3)
        for intervencion in efecto["intervenciones"]:
            self.assertEqual(len(intervencion["caida"]), pasos)
            self.assertTrue(0 <= intervencion["atencion_recibida"] <= 1)
        todas = {i["zona"]: i for i in efecto["intervenciones"] if i["capas"] == "todas"}
        for zona in ("voz", "texto"):
            self.assertAlmostEqual(todas[zona]["atencion_recibida"], efecto["atencion_media"][zona], places=3)
            self.assertGreaterEqual(intervencion["kl_media"], -1e-4)
        self.assertEqual([i["rango"] for i in efecto["intervenciones"][:4]],
                         [[1, 4], [1, 2], [3, 3], [4, 4]])

    def test_mascara_neutra_no_cambia_nada(self):
        def calcular():
            entrada, _, prefijo = self.explicador._preparar(*self.args)
            unos = self.explicador.torch.ones(4, 1, 4, 1, 1)
            return (self.explicador._logaritmos(entrada, prefijo),
                    self.explicador._logaritmos(entrada, prefijo, unos))
        base, con = self.explicador._con_cerrojo(calcular)
        np.testing.assert_allclose(base.numpy(), con.numpy(), atol=1e-5)

    def test_las_intervenciones_cambian_la_distribucion(self):
        # Con pesos aleatorios el efecto es pequeño, pero debe existir y solo donde toca.
        def calcular():
            torch = self.explicador.torch
            entrada, _, prefijo = self.explicador._preparar(*self.args)
            largo = entrada.shape[1]
            base = self.explicador._logaritmos(entrada, prefijo)
            cambios = {}
            for zona in ("voz", "texto"):
                mascara = self.explicador._mascara_zona(largo, prefijo, zona, None)
                self.assertEqual(tuple(mascara.shape), (4, 1, 1, largo, largo))
                inicio, fin = (0, 32) if zona == "voz" else (32, prefijo - 1)
                self.assertEqual(float(mascara[0, 0, 0, prefijo - 1:, inicio:fin].abs().max()), 0)
                self.assertEqual(float(mascara[0, 0, 0].sum()), largo * largo - (largo - prefijo + 1) * (fin - inicio))
                cambios[zona] = float((base - self.explicador._logaritmos(entrada, prefijo, mascara)).abs().max())
            tramo = self.explicador._mascara_zona(largo, prefijo, "voz", [3])
            self.assertEqual(float(tramo[:3].min()), 1)
            self.assertEqual(float(tramo[3, 0, 0, prefijo - 1:, :32].max()), 0)
            apagada = torch.ones(4, 1, 4, 1, 1)
            apagada[1, 0, 2] = 0
            cambios["cabeza"] = float((base - self.explicador._logaritmos(entrada, prefijo, apagada)).abs().max())
            return cambios
        for nombre, cambio in self.explicador._con_cerrojo(calcular).items():
            self.assertGreater(cambio, 1e-5, nombre)

    def test_efecto_cabezas(self):
        efecto = self.explicador.efecto_cabezas(*self.args, 1)
        self.assertEqual(efecto["capa"], 2)
        self.assertEqual([c["cabeza"] for c in efecto["cabezas"]], [1, 2, 3, 4])

    def test_rechazos(self):
        with self.assertRaises(ValueError):
            self.explicador.atencion(*self.args, 9)
        with self.assertRaises(ValueError):
            self.explicador.atencion(*self.args, 0, 7)
        with self.assertRaises(ValueError):
            self.explicador.efecto_causal(self.vectores, self.resultado.texto_ids, [])
        with self.generador.cerrojo, self.assertRaises(GeneradorOcupado):
            self.explicador.efecto_causal(*self.args)

    def test_matriz_cuantizada_usa_un_solo_maximo(self):
        matriz = np.array([[0.0, 0.5], [0.25, 1.0]], np.float32)
        c = matriz_cuantizada(matriz)
        q = np.frombuffer(base64.b64decode(c["q"]), np.uint8).reshape(c["filas"], c["columnas"])
        np.testing.assert_allclose(q / 255 * c["max"], matriz, atol=1 / 255)


if __name__ == "__main__":
    unittest.main()
