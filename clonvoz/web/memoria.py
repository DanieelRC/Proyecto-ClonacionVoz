"""Puente temporal entre etapas: guarda el mel, nunca el WAV ni sus muestras.

Cada entrada pertenece a una sesión del navegador. Caduca a los diez minutos
o al descartarla. El límite de ocho entradas evita crecimiento sin control.
Los temporizadores borran los datos aunque no lleguen nuevas solicitudes.
"""

from collections import OrderedDict
from dataclasses import dataclass
import threading

import numpy as np


@dataclass
class MelGuardado:
    propietario: str
    id_corrida: str
    nombre: str
    mel: np.ndarray
    duracion_s: float
    advertencias: list[str]


class MemoriaMel:
    def __init__(self, caducidad_s=600, capacidad=8):
        self.caducidad_s = caducidad_s
        self.capacidad = capacidad
        self.entradas = OrderedDict()
        self.cerrojo = threading.Lock()

    def guardar(self, propietario, nombre, analisis):
        """Conserva la matriz calculada por etapa 1, sin recalcularla.

        1. Separar el mel y los metadatos del resto del análisis.
        2. Desalojar la entrada más antigua si ya alcanzamos el límite.
        3. Programar la eliminación automática; el temporizador no impide salir.
        """
        entrada = MelGuardado(propietario, analisis.id_corrida, nombre, analisis.mel,
                              analisis.audio.duracion_s, list(analisis.validacion.advertencias))
        temporizador = threading.Timer(self.caducidad_s, self.descartar,
                                       args=(propietario, entrada.id_corrida))
        temporizador.daemon = True
        with self.cerrojo:
            while len(self.entradas) >= self.capacidad:
                _, (_, anterior) = self.entradas.popitem(last=False)
                anterior.cancel()
            self.entradas[entrada.id_corrida] = (entrada, temporizador)
            temporizador.start()

    def obtener(self, propietario, identificador):
        """Entrega únicamente entradas de esta sesión; no renueva la caducidad."""
        with self.cerrojo:
            registro = self.entradas.get(identificador)
            if registro is None or registro[0].propietario != propietario:
                raise KeyError(identificador)
            return registro[0]

    def descartar(self, propietario, identificador):
        """Libera un resultado reemplazado, vencido o descartado explícitamente."""
        with self.cerrojo:
            registro = self.entradas.get(identificador)
            if registro is not None and registro[0].propietario == propietario:
                del self.entradas[identificador]
                registro[1].cancel()
