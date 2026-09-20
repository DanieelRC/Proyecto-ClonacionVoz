"""Casos límite de visualización; no necesitan descargar el modelo."""
import unittest
from unittest.mock import patch
import numpy as np
from clonvoz.identidad import Identidad
from clonvoz.graficas_identidad import renderizar_identidades, similitud_consultas


class PruebasVistas(unittest.TestCase):
    def test_coseno_y_consulta_nula(self):
        x = np.array([[1., 0], [-1, 0], [0, 1], [0, 0]])
        resultado = similitud_consultas(x)
        np.testing.assert_allclose(resultado[:3, :3], [[1, -1, 0], [-1, 1, 0], [0, 0, 1]])
        self.assertTrue(np.isnan(resultado[3]).all())

    def test_cero_y_pesos_menores_al_antiguo_limite(self):
        from matplotlib.colors import PowerNorm
        for peso in (0., 1e-20):
            with self.subTest(peso=peso):
                vectores = np.arange(32*1024, dtype=np.float32).reshape(32, 1024)
                copia = vectores.copy()
                atencion = np.full((32, 8), peso, dtype=np.float32)
                propias = np.full((1, 32, 32), 1/32, dtype=np.float32)
                r = Identidad('prueba', vectores, atencion, propias,
                              [{'inicio_s': 0, 'fin_s': 1, 'frames': 8, 'columna_inicio': 0}],
                              {}, [], 'cpu')
                with patch('clonvoz.graficas_identidad.PowerNorm', wraps=PowerNorm) as norma:
                    imagenes = renderizar_identidades([r])[0]
                    if peso:
                        self.assertTrue(norma.called)
                        self.assertAlmostEqual(norma.call_args.kwargs['vmax']/peso, 1., places=6)
                    else:
                        norma.assert_not_called()
                self.assertEqual(set(imagenes), {'atencion', 'atencion_inicial', 'similitud', 'centrada', 'identidad'})
                # renderizar_identidades entrega los bytes del PNG; envolverlos
                # como data URI es trabajo de la capa web. Comprobar la firma del
                # formato es más estricto que comprobar un prefijo de texto.
                self.assertTrue(all(i.startswith(b'\x89PNG\r\n\x1a\n')
                                    for v in imagenes.values() for i in v.values()))
                np.testing.assert_array_equal(vectores, copia)


if __name__ == '__main__':
    unittest.main()
