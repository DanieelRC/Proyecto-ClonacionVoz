"""Corridas completas grabadas: el formato que reproduce la página sin modelo.

El pipeline emite objetos (Analisis, Identidad, Sintesis); aquí se traducen al
JSON que la página pinta, con las figuras y el audio ya dentro. El servidor usa
la misma traducción para el canal WebSocket, y la CLI para grabar un respaldo:
una corrida en vivo y una grabada son la misma secuencia de eventos.
"""

from __future__ import annotations

import base64
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from . import graficas
from .fases import a_wav
from .graficas_identidad import renderizar_identidades
from .graficas_locutores import renderizar_identificacion
from .graficas_sintesis import renderizar_sintesis

FORMATO = "clonvoz-corrida-completa"
NOMBRE_VALIDO = re.compile(r"^[\w\-. ]{1,120}\.json$")


def figuras_como_uri(por_figura) -> dict:
    return {nombre: {tema: graficas.como_data_uri(png) for tema, png in temas.items()}
            for nombre, temas in por_figura.items()}


def wav_como_data_uri(muestras, sr) -> str:
    return "data:audio/wav;base64," + base64.b64encode(a_wav(muestras, sr=sr)).decode("ascii")


def respuesta_sintesis(sintesis, mel_referencia) -> dict:
    """Sintesis -> lo que la página necesita: cifras, figuras y el audio."""
    datos = sintesis.como_dict()
    datos["graficas"] = figuras_como_uri(renderizar_sintesis(sintesis, mel_referencia))
    datos["audio"] = wav_como_data_uri(sintesis.onda, sintesis.sr)
    return datos


def respuesta_locutor(identificacion, archivo) -> dict:
    """Identificacion -> el nombre, los candidatos y la proyección con el audio marcado."""
    datos = identificacion.como_dict()
    datos["archivo"] = archivo
    datos["graficas"] = figuras_como_uri(renderizar_identificacion(identificacion))
    return datos


class Traductor:
    """Convierte cada evento del pipeline en el JSON que pinta la página."""

    def __init__(self, nombre_archivo, texto, id_corrida):
        self.nombre = nombre_archivo
        self.texto = texto.strip()
        self.id_corrida = id_corrida
        self.mel = None

    def __call__(self, tipo, datos):
        if tipo == "etapa1":
            self.mel = datos.mel
            salida = datos.como_dict()
            salida["archivo"] = self.nombre
            salida["graficas"] = figuras_como_uri(graficas.renderizar(datos))
            return salida
        if tipo == "etapa2":
            resumen = datos.como_dict()
            resumen["archivo"] = self.nombre
            resumen["graficas"] = figuras_como_uri(renderizar_identidades([datos])[0])
            return {"resultados": [resumen]}
        if tipo == "locutor":
            return respuesta_locutor(datos, self.nombre)
        if tipo == "etapa4":
            salida = respuesta_sintesis(datos, self.mel)
            salida["id_corrida"] = self.id_corrida
            salida["texto"] = self.texto
            return salida
        return datos


class Grabador:
    """Acumula los eventos con su tiempo y los guarda en el formato de la página."""

    def __init__(self):
        self.inicio = time.perf_counter()
        self.eventos = []

    def __call__(self, tipo, datos):
        self.eventos.append({"tipo": tipo, "t": round((time.perf_counter() - self.inicio) * 1000),
                             "datos": datos})

    def como_dict(self) -> dict:
        return {"formato": FORMATO, "version": 1, "guardada": datetime.now(timezone.utc).isoformat(),
                "eventos": self.eventos}

    def guardar(self, ruta):
        Path(ruta).write_text(json.dumps(self.como_dict(), ensure_ascii=False), encoding="utf-8")


def ruta_de_grabacion(carpeta, nombre) -> Path:
    """Valida el nombre y devuelve su ruta dentro de `carpeta`; KeyError si no es válido."""
    if not isinstance(nombre, str) or not NOMBRE_VALIDO.match(nombre) or nombre.startswith("."):
        raise KeyError(nombre)
    carpeta = Path(carpeta).resolve()
    ruta = (carpeta / nombre).resolve()
    if ruta.parent != carpeta or not ruta.is_file():
        raise KeyError(nombre)
    return ruta


def listar_grabaciones(carpeta) -> list[dict]:
    """Las corridas guardadas en `carpeta`, la más reciente primero."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return []
    archivos = [a for a in carpeta.glob("*.json") if a.is_file() and NOMBRE_VALIDO.match(a.name)]
    archivos.sort(key=lambda a: a.stat().st_mtime, reverse=True)
    return [{"nombre": a.name, "kb": round(a.stat().st_size / 1024),
             "modificada": datetime.fromtimestamp(a.stat().st_mtime, timezone.utc).isoformat()}
            for a in archivos]
