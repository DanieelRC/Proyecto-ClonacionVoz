"""Figuras de la etapa 4: los tres mels comparables y la forma de onda de salida."""

from __future__ import annotations

import numpy as np

from . import config
from .graficas import (ANCHO_COMPLETO, MAPAS_CALOR, TEMAS, _a_png, _barra_de_escala, _etiquetar,
                       _figura, _titular, _vestir_eje)

COLUMNAS_ONDA = 1400


def grafica_mels(mel_referencia, mel_dvae, mel_salida, tema_nombre="claro") -> bytes:
    """Mel de la referencia, mel que describen los tokens (DVAE) y mel de la voz generada.

    Los tres comparten la escala de color y los segundos por unidad de ancho, así
    que se pueden comparar a ojo. La referencia dice otro texto: se compara el
    timbre y el rango, no el contenido.
    """
    tema = TEMAS[tema_nombre]
    paneles = [("Referencia (etapa 1) — otro texto, misma voz", mel_referencia),
               ("Lo que describen los tokens (DVAE) — rama didáctica", mel_dvae),
               ("Voz generada (HiFi-GAN) — el resultado real", mel_salida)]
    paneles = [(titulo, np.asarray(mel)) for titulo, mel in paneles if mel is not None]
    minimo = min(float(mel.min()) for _, mel in paneles)
    maximo = max(float(mel.max()) for _, mel in paneles)
    segundos = [mel.shape[1] * config.HOP_LENGTH / config.SR for _, mel in paneles]

    figura = _figura(tema, alto=3.0 * len(paneles))
    ejes = figura.subplots(len(paneles), 1, squeeze=False)[:, 0]
    imagen = None
    for eje, (titulo, mel), duracion in zip(ejes, paneles, segundos):
        _vestir_eje(eje, tema, con_rejilla=False)
        imagen = eje.imshow(mel, origin="lower", aspect="auto", cmap=MAPAS_CALOR[tema_nombre],
                            vmin=minimo, vmax=maximo, extent=(0.0, duracion, 0.0, mel.shape[0]),
                            interpolation="nearest")
        eje.set_xlim(0.0, max(segundos))
        _titular(eje, tema, f"{titulo} · [{mel.shape[0]}, {mel.shape[1]}]")
        _etiquetar(eje, tema, y="canal Mel")
    _etiquetar(ejes[-1], tema, x="tiempo (s)")
    _barra_de_escala(figura, imagen, list(ejes), tema, "log de energía Mel (misma escala)")
    return _a_png(figura)


def grafica_onda(onda, sr, tema_nombre="claro") -> bytes:
    """Forma de onda de la voz generada."""
    tema = TEMAS[tema_nombre]
    onda = np.asarray(onda, dtype=np.float32)
    figura = _figura(tema, alto=2.9, ancho=ANCHO_COMPLETO)
    eje = figura.add_subplot(111)
    _vestir_eje(eje, tema)

    n = onda.shape[0]
    if n > COLUMNAS_ONDA:
        por_columna = n // COLUMNAS_ONDA
        bloques = onda[:por_columna * COLUMNAS_ONDA].reshape(COLUMNAS_ONDA, por_columna)
        tiempo = np.arange(COLUMNAS_ONDA) * por_columna / sr
        minimos, maximos = bloques.min(axis=1), bloques.max(axis=1)
    else:
        tiempo, minimos, maximos = np.arange(n) / sr, onda, onda
    eje.fill_between(tiempo, minimos, maximos, color=tema["salida"], linewidth=0.0)
    eje.axhline(0.0, color=tema["rejilla"], linewidth=0.8)
    limite = max(float(np.max(np.abs(onda))) * 1.1 if n else 0.0, 0.01)
    eje.set_ylim(-limite, limite)
    eje.set_xlim(0.0, max(n / sr, 1e-3))
    _titular(eje, tema, f"Voz generada — {n:,} muestras a {sr:,} Hz".replace(",", " "))
    _etiquetar(eje, tema, x="tiempo (s)", y="amplitud")
    return _a_png(figura)


def renderizar_sintesis(sintesis, mel_referencia, temas=("claro", "oscuro")) -> dict:
    """Las dos figuras de la etapa 4 en ambos temas."""
    return {
        "mels": {t: grafica_mels(mel_referencia, sintesis.mel_dvae, sintesis.mel_salida, t) for t in temas},
        "onda": {t: grafica_onda(sintesis.onda, sintesis.sr, t) for t in temas},
    }
