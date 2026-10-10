"""Etapa 5: las cuatro etapas en una sola corrida que avisa de cada paso.

Ningún módulo se reescribe: se llaman, en orden, los mismos objetos que usan las
etapas por separado. Lo nuevo es `emitir`, que recibe un evento en el momento en
que ocurre cada cosa. Quien lo recibe (el servidor, la línea de comandos) decide
cómo mostrarlo; aquí no hay Flask, ni JSON, ni figuras.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .analisis import analizar
from .explicacion import Explicador
from .generacion import evento_paso
from .identidad import ModeloNoDisponible, identidad_desde_mel

BLOQUES = (
    ("audio", "Audio de referencia", "audio"),
    ("mel", "Mel-espectrograma", "audio"),
    ("perceiver", "Perceiver", "audio"),
    ("gpt", "GPT-2", "generacion"),
    ("tokens", "Tokens de audio", "generacion"),
    ("dvae", "DVAE", "generacion"),
    ("hifigan", "HiFi-GAN", "salida"),
    ("salida", "Audio clonado", "salida"),
)
"""(id, nombre, ruta de color) de los ocho bloques del diagrama, en orden."""


class CorridaCancelada(Exception):
    """Se pidió detener la corrida entre dos bloques."""


@dataclass
class Corrida:
    """Todo lo que produjo una corrida completa."""

    analisis: object
    identidad: object
    generacion: object
    sintesis: object
    causal: dict | None


def ejecutar_corrida(fuente, texto, modelos, emitir, *, detener=None, determinista=False,
                     semilla=0, metodo_f0="pyin", explicar=True, id_corrida=None) -> Corrida:
    """Audio de referencia + texto -> voz clonada, emitiendo cada paso.

    `modelos` entrega los modelos ya cargados: `identidad()`, `generador()`,
    `sintetizador()` y `fases()` (el DVAE; puede fallar si falta dvae.pth).
    `emitir(tipo, datos)` recibe:

        bloque   {id, estado: activo|listo|omitido, ms, forma, nota}
        etapa1   Analisis          etapa2   Identidad
        inicio / paso / fin        los mismos eventos de la etapa 3
        etapa4   Sintesis          causal   dict de Explicador.efecto_causal

    1. Audio y mel (etapa 1).            4. DVAE: el mel que describen los tokens.
    2. Perceiver (etapa 2).              5. Latentes + HiFi-GAN: la voz (etapa 4).
    3. GPT-2, token a token (etapa 3).   6. Intervenciones causales, ya con el audio listo.
    """
    def revisar():
        if detener is not None and detener.is_set():
            raise CorridaCancelada()

    def activo(bloque, nota=None):
        revisar()
        emitir("bloque", {"id": bloque, "estado": "activo", "nota": nota})
        return time.perf_counter()

    def listo(bloque, inicio=None, ms=None, forma=None, nota=None, estado="listo"):
        if ms is None and inicio is not None:
            ms = (time.perf_counter() - inicio) * 1000
        emitir("bloque", {"id": bloque, "estado": estado, "ms": None if ms is None else round(ms, 1),
                          "forma": forma, "nota": nota})

    # 1. Etapa 1: cargar, validar y calcular las cuatro características.
    activo("audio")
    analisis = analizar(fuente, metodo_f0=metodo_f0, id_corrida=id_corrida)
    listo("audio", ms=analisis.tiempos_ms["carga"], forma=[analisis.audio.n_muestras],
          nota=f"{analisis.audio.duracion_s:.1f} s a {analisis.audio.sr} Hz")
    activo("mel")
    listo("mel", ms=analisis.tiempos_ms["mel"], forma=list(analisis.mel.shape))
    emitir("etapa1", analisis)

    # 2. Etapa 2: el mel completo pasa por el Perceiver.
    inicio = activo("perceiver", "cargando el modelo la primera vez")
    identidad = identidad_desde_mel(analisis.mel, modelos.identidad(), analisis.id_corrida,
                                    analisis.audio.duracion_s, analisis.validacion.advertencias)
    listo("perceiver", ms=identidad.tiempos_ms["codificacion"], forma=list(identidad.vectores.shape))
    emitir("etapa2", identidad)

    # 3. Etapa 3: el GPT-2 genera; cada token es un evento.
    inicio = activo("gpt", "cargando el modelo la primera vez")
    generador = modelos.generador()
    ids, piezas = generador.tokenizar(texto)
    emitir("inicio", generador.evento_inicio(analisis.id_corrida, texto, ids, piezas, determinista, semilla))
    emitir("bloque", {"id": "gpt", "estado": "activo", "nota": f"{len(ids)} fichas de texto"})
    emitir("bloque", {"id": "tokens", "estado": "activo", "nota": None})
    generacion = generador.generar(identidad.vectores, texto, detener=detener,
                                   al_paso=lambda paso: emitir("paso", evento_paso(paso)),
                                   determinista=determinista, semilla=semilla)
    emitir("fin", generacion.como_dict())
    listo("gpt", ms=generacion.ms_total, nota=f"{generacion.ms_por_token:.0f} ms por token")
    listo("tokens", forma=[int(len(generacion.tokens))], nota=generacion.motivo)
    if generacion.motivo == "cancelada":
        raise CorridaCancelada()

    # 4. DVAE: rama didáctica. Si falta dvae.pth la corrida sigue.
    inicio = activo("dvae")
    try:
        fases = modelos.fases()
        mel_dvae = fases.mel_desde_tokens(generacion.tokens)
        listo("dvae", inicio, forma=list(mel_dvae.shape), nota="no alimenta al vocoder")
    except (ModeloNoDisponible, ValueError) as exc:
        fases = None
        listo("dvae", inicio, nota=str(exc), estado="omitido")

    # 5. Etapa 4: latentes del GPT-2 + vector de hablante -> HiFi-GAN.
    inicio = activo("hifigan", "cargando el vocoder la primera vez")
    sintetizador = modelos.sintetizador()
    embedding = sintetizador.embedding_hablante(analisis.audio.muestras)
    sintesis = sintetizador.sintetizar(identidad.vectores, ids, generacion.tokens, embedding, fases)
    listo("hifigan", inicio, forma=list(sintesis.latentes.shape), nota="latentes del GPT-2 → onda")
    activo("salida")
    listo("salida", ms=0.0, forma=[int(len(sintesis.onda))],
          nota=f"{sintesis.duracion_s:.1f} s a {sintesis.sr} Hz")
    emitir("etapa4", sintesis)

    # 6. Evidencia causal: después del audio, para no retrasar la voz.
    causal = None
    if explicar and not (detener is not None and detener.is_set()):
        emitir("explicando", {})
        causal = Explicador(generador).efecto_causal(identidad.vectores, ids, generacion.tokens)
        emitir("causal", causal)

    return Corrida(analisis, identidad, generacion, sintesis, causal)
