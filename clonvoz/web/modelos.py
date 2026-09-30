"""Registro de modelos cargados, compartido entre las etapas del servidor.

Cada modelo se construye una sola vez, la primera vez que una etapa lo pide. Si
la construcción falla no queda nada guardado, así que el siguiente intento
vuelve a probar (por ejemplo, después de descargar un archivo que faltaba).
"""

import threading


class Modelos:
    def __init__(self):
        self._instancias = {}
        # Reentrante: la fábrica de un modelo puede pedir otro (las fases piden la etapa 2).
        self._cerrojo = threading.RLock()

    def obtener(self, nombre, fabrica):
        """Devuelve el modelo `nombre`, construyéndolo con `fabrica` si aún no existe."""
        with self._cerrojo:
            if nombre not in self._instancias:
                self._instancias[nombre] = fabrica()
            return self._instancias[nombre]
