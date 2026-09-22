"""Almacenamiento temporal en memoria de resultados de mel para la etapa 2."""

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
    """Gestiona en memoria los mel calculados por sesión con caducidad y límite de capacidad."""

    def __init__(self, caducidad_s=600, capacidad=8):
        self.caducidad_s = caducidad_s
        self.capacidad = capacidad
        self.entradas = OrderedDict()
        self.cerrojo = threading.Lock()

    def guardar(self, propietario, nombre, analisis):
        """Guarda el resultado del mel y programa su caducidad automática."""
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
        """Recupera la entrada correspondiente al propietario e identificador."""
        with self.cerrojo:
            registro = self.entradas.get(identificador)
            if registro is None or registro[0].propietario != propietario:
                raise KeyError(identificador)
            return registro[0]

    def descartar(self, propietario, identificador):
        """Elimina una entrada y cancela su temporizador."""
        with self.cerrojo:
            registro = self.entradas.get(identificador)
            if registro is not None and registro[0].propietario == propietario:
                del self.entradas[identificador]
                registro[1].cancel()
