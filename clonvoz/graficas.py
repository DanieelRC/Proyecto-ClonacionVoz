"""
Render de las cuatro visualizaciones a PNG.

Decisiones que vale la pena conocer antes de tocar este archivo:

* **Backend Agg y API orientada a objetos.** No se usa `pyplot` en ningún punto:
  `pyplot` mantiene estado global de figuras y no es seguro entre hilos, y el
  servidor de Flask atiende peticiones en hilos. Se crean `Figure` y
  `FigureCanvasAgg` explícitamente, que sí lo es.

* **Dos temas, renderizados juntos.** Cada figura se dibuja en su variante clara y
  oscura en la misma corrida (RNF-07). Volver a graficar al cambiar de tema daría
  un salto perceptible; graficar es barato y así el cambio es un fundido (RNF-06).

* **El F0 va en su propio panel, no superpuesto en un segundo eje y.** Poner
  amplitud y hercios en dos escalas sobre el mismo plano es el error clásico de
  visualización: la alineación entre ambas escalas es arbitraria e inventa una
  correlación que no está en los datos. Los dos paneles comparten el eje de
  tiempo, que es lo que hace ver "cómo sube y baja la entonación" sin mentir.

* **Paleta validada, no elegida a ojo.** Los cuatro colores de ruta se
  verificaron con el validador de paletas: separación bajo daltonismo y con
  visión normal en ambos modos, con todos los pares en juego. La primera
  propuesta (teal, ámbar, violeta, verde) falló —teal y verde resultaron
  indistinguibles incluso con visión normal— y se sustituyó. Los valores viven
  también en `static/css/tokens.css`: **si cambias unos, cambia los otros.**

* **La escala de los mapas de calor es magma.** Es la convención del campo y una
  rampa perceptualmente uniforme; se usa como "calor semántico" (energía), y por
  eso cada mapa lleva su barra de escala.
"""

from __future__ import annotations

import io
import textwrap
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # antes de cualquier import de matplotlib que cree figuras

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from . import config
from .analisis import Analisis

def _familia_disponible(candidatas: tuple[str, ...]) -> str:
    """
    Primera tipografía de la lista que matplotlib tenga instalada.

    Hace falta resolverla una sola vez, y no dejarle la lista a matplotlib: al
    pasarle una familia que no existe, avisa por cada texto que dibuja, y una
    figura con ejes y barra de escala son decenas de textos. En Linux, donde no
    hay Segoe UI, eso llenaba la terminal de cientos de líneas de aviso.
    """
    from matplotlib import font_manager

    instaladas = {fuente.name for fuente in font_manager.fontManager.ttflist}
    for nombre in candidatas:
        if nombre in instaladas:
            return nombre
    return "DejaVu Sans"  # la que matplotlib trae siempre consigo


FAMILIA = _familia_disponible(("Segoe UI", "Verdana", "DejaVu Sans"))
"""
Familia tipográfica de las figuras.

Mantiene una sola tipografía entre la página y las imágenes (RNF-04): en Windows
resuelve a Segoe UI, la misma que usa la interfaz. En otros sistemas cae en la
tipografía propia de matplotlib.
"""

TAMANOS = {"titulo": 15.0, "etiqueta": 12.5, "tick": 11.5, "nota": 10.5}
"""Tamaños de fuente, generosos para que las figuras se lean proyectadas (RNF-08)."""

TEMAS = {
    "claro": {
        "superficie": "#ffffff",
        "texto": "#0f172a",
        "texto_suave": "#52606d",
        "rejilla": "#e5e9ef",
        "audio": "#2a78d6",
        "umbral": "#8794a7",
    },
    "oscuro": {
        "superficie": "#0f172a",
        "texto": "#f1f5f9",
        "texto_suave": "#9aa8bd",
        "rejilla": "#22304a",
        "audio": "#3987e5",
        "umbral": "#7d8ca3",
    },
}
"""
Colores de las figuras por tema.

El modo oscuro no es una inversión automática del claro: el acento tiene su
propio paso para la superficie oscura, validado contra ella.
"""

MAPA_CALOR = "magma"
COLUMNAS_ENVOLVENTE = 1400
"""Columnas en las que se resume la forma de onda (ver `_envolvente`)."""

DPI = 110

ANCHO_COMPLETO = 10.0
"""Pulgadas de las figuras que ocupan todo el ancho (forma de onda y F0)."""

ANCHO_MEDIO = 6.6
"""
Pulgadas de las figuras que van a media columna (los dos mapas de calor).

No es un capricho de composición: una figura de 10 pulgadas mostrada en una
columna de 660 píxeles se reduce a dos tercios, y con ella el tamaño aparente de
los ejes y las etiquetas, que es justo lo que RNF-08 pide conservar. Dibujarlas
más angostas mantiene el texto al tamaño con el que fue pensado.
"""

RANGO_DINAMICO_DB = 80.0
"""Decibelios visibles por debajo del pico en el espectrograma STFT."""


def _envolvente(
    amplitud: np.ndarray, columnas: int = COLUMNAS_ENVOLVENTE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resume la señal en `columnas` pares (mínimo, máximo).

    Un audio de 10 s son 220 500 muestras y la figura mide unos 1 100 píxeles de
    ancho: dibujar punto por punto sería lento y produciría un bloque sólido en
    vez de una forma de onda. Tomar el mínimo y el máximo de cada bloque conserva
    la silueta real —los picos no se pierden, como sí pasaría al submuestrear— y
    es lo que se dibuja como banda rellena.
    """
    n = amplitud.shape[0]
    if n == 0:
        vacio = np.zeros(0, dtype=np.float32)
        return vacio, vacio, vacio
    if n <= columnas:
        tiempo = np.arange(n, dtype=np.float32) / config.SR
        return tiempo, amplitud, amplitud

    por_columna = n // columnas
    utiles = por_columna * columnas
    bloques = amplitud[:utiles].reshape(columnas, por_columna)
    minimos = bloques.min(axis=1)
    maximos = bloques.max(axis=1)
    tiempo = (np.arange(columnas, dtype=np.float32) * por_columna) / config.SR
    return tiempo, minimos, maximos


def _figura(tema: dict, alto: float, ancho: float = ANCHO_COMPLETO) -> Figure:
    figura = Figure(figsize=(ancho, alto), dpi=DPI, layout="constrained")
    figura.set_facecolor(tema["superficie"])
    FigureCanvasAgg(figura)
    return figura


def _vestir_eje(eje, tema: dict, con_rejilla: bool = True) -> None:
    """Aplica el tema al eje: cromo recesivo, texto legible, sin adornos."""
    eje.set_facecolor(tema["superficie"])
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_color(tema["rejilla"])
        eje.spines[lado].set_linewidth(0.8)
    eje.tick_params(
        colors=tema["texto_suave"], labelsize=TAMANOS["tick"], width=0.8, length=4
    )
    for etiqueta in eje.get_xticklabels() + eje.get_yticklabels():
        etiqueta.set_fontfamily(FAMILIA)
    if con_rejilla:
        # Líneas sólidas y de un solo tono sobre la superficie: el punteado
        # agrega ruido y se lee como "umbral" cuando solo es una rejilla.
        eje.grid(True, axis="y", color=tema["rejilla"], linewidth=0.8, alpha=1.0)
        eje.set_axisbelow(True)


def _titular(eje, tema: dict, titulo: str, ancho: float = ANCHO_COMPLETO) -> None:
    """
    Pone el título, partiéndolo en varias líneas si no cabe a lo ancho.

    matplotlib no acomoda los títulos largos: ni los encoge ni los parte, los
    dibuja hasta salirse del lienzo y el PNG sale con el texto cortado. Como las
    figuras de media columna miden dos tercios de las anchas, el corte aparecía
    solo en ellas. El presupuesto de caracteres se estima del ancho en pulgadas.
    """
    presupuesto = max(24, int(ancho * 7.2))
    eje.set_title(
        "\n".join(textwrap.wrap(titulo, presupuesto)) or titulo,
        loc="left",
        color=tema["texto"],
        fontsize=TAMANOS["titulo"],
        fontfamily=FAMILIA,
        pad=12,
    )


def _etiquetar(eje, tema: dict, x: str | None = None, y: str | None = None) -> None:
    if x:
        eje.set_xlabel(x, color=tema["texto_suave"], fontsize=TAMANOS["etiqueta"], fontfamily=FAMILIA)
    if y:
        eje.set_ylabel(y, color=tema["texto_suave"], fontsize=TAMANOS["etiqueta"], fontfamily=FAMILIA)


def _barra_de_escala(figura: Figure, imagen, eje, tema: dict, etiqueta: str) -> None:
    """Barra de color: es la leyenda de escala que toda rampa continua necesita."""
    barra = figura.colorbar(imagen, ax=eje, pad=0.015, fraction=0.046)
    barra.outline.set_visible(False)
    barra.ax.tick_params(colors=tema["texto_suave"], labelsize=TAMANOS["tick"], width=0.8, length=3)
    for etiqueta_tick in barra.ax.get_yticklabels():
        etiqueta_tick.set_fontfamily(FAMILIA)
    barra.set_label(etiqueta, color=tema["texto_suave"], fontsize=TAMANOS["nota"])
    barra.ax.yaxis.label.set_fontfamily(FAMILIA)


def _a_png(figura: Figure) -> bytes:
    buffer = io.BytesIO()
    figura.savefig(buffer, format="png", facecolor=figura.get_facecolor())
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Las cuatro figuras
# --------------------------------------------------------------------------


def grafica_forma_de_onda(analisis: Analisis, tema_nombre: str = "claro") -> bytes:
    """Amplitud contra tiempo: la señal tal como está guardada en el archivo."""
    tema = TEMAS[tema_nombre]
    figura = _figura(tema, alto=2.9)
    eje = figura.add_subplot(111)
    _vestir_eje(eje, tema)

    tiempo, minimos, maximos = _envolvente(analisis.amplitud)
    eje.fill_between(tiempo, minimos, maximos, color=tema["audio"], linewidth=0.0)
    eje.axhline(0.0, color=tema["rejilla"], linewidth=0.8)

    limite = max(float(np.max(np.abs(analisis.amplitud))) * 1.1, 0.01)
    eje.set_ylim(-limite, limite)
    eje.set_xlim(0.0, analisis.audio.duracion_s)

    _titular(
        eje,
        tema,
        f"Forma de onda — {analisis.audio.n_muestras:,} muestras "
        f"a {config.SR:,} Hz".replace(",", " "),
    )
    _etiquetar(eje, tema, x="tiempo (s)", y="amplitud")
    return _a_png(figura)


def grafica_espectrograma(analisis: Analisis, tema_nombre: str = "claro") -> bytes:
    """Mapa de calor de frecuencia contra tiempo, en decibelios."""
    tema = TEMAS[tema_nombre]
    figura = _figura(tema, alto=3.5, ancho=ANCHO_MEDIO)
    eje = figura.add_subplot(111)
    _vestir_eje(eje, tema, con_rejilla=False)

    alto, ancho = analisis.espectrograma_db.shape
    imagen = eje.imshow(
        analisis.espectrograma_db,
        origin="lower",
        aspect="auto",
        cmap=MAPA_CALOR,
        vmin=-RANGO_DINAMICO_DB,
        vmax=0.0,
        extent=(0.0, analisis.audio.duracion_s, 0.0, config.SR / 2),
        interpolation="nearest",
    )

    # Umbral real, no rejilla: hasta aquí llega el banco de filtros Mel. Ayuda a
    # explicar por qué el mel de la derecha no ve todo lo que se ve aquí.
    eje.axhline(config.MEL_FMAX, color=tema["umbral"], linewidth=1.2, linestyle=(0, (5, 4)))
    eje.text(
        analisis.audio.duracion_s * 0.995,
        config.MEL_FMAX + config.SR / 2 * 0.02,
        f"fmax del mel · {config.MEL_FMAX // 1000} kHz",
        color=tema["umbral"],
        fontsize=TAMANOS["nota"],
        fontfamily=FAMILIA,
        ha="right",
        va="bottom",
    )

    _titular(eje, tema, f"Espectrograma STFT — [{alto}, {ancho}]", ancho=ANCHO_MEDIO)
    _etiquetar(eje, tema, x="tiempo (s)", y="frecuencia (Hz)")
    _barra_de_escala(figura, imagen, eje, tema, "dB relativos al pico")
    return _a_png(figura)


def grafica_mel(analisis: Analisis, tema_nombre: str = "claro") -> bytes:
    """
    El mel-espectrograma de 80 canales: el tensor que recibe la etapa 2.

    Se dibuja con la misma paleta y el mismo eje temporal que el espectrograma
    STFT, para que al verlos uno junto al otro se note la diferencia entre ambas
    representaciones, que es exactamente lo que pide el documento de etapas.
    """
    tema = TEMAS[tema_nombre]
    figura = _figura(tema, alto=3.5, ancho=ANCHO_MEDIO)
    eje = figura.add_subplot(111)
    _vestir_eje(eje, tema, con_rejilla=False)

    canales, frames = analisis.mel.shape
    imagen = eje.imshow(
        analisis.mel,
        origin="lower",
        aspect="auto",
        cmap=MAPA_CALOR,
        extent=(0.0, analisis.audio.duracion_s, 0.0, canales),
        interpolation="nearest",
    )

    _titular(
        eje,
        tema,
        f"Mel-espectrograma — [{canales}, {frames}] · entrada del Perceiver (etapa 2)",
        ancho=ANCHO_MEDIO,
    )
    _etiquetar(eje, tema, x="tiempo (s)", y=f"canal Mel (0 Hz – {config.MEL_FMAX // 1000} kHz)")
    _barra_de_escala(figura, imagen, eje, tema, "log de energía Mel")
    return _a_png(figura)


def grafica_f0(analisis: Analisis, tema_nombre: str = "claro") -> bytes:
    """
    Contorno de F0 junto a la forma de onda, en dos paneles con eje de tiempo común.

    Los tramos sin voz quedan como huecos en la línea: dibujarlos en cero haría
    ver caídas de entonación que no ocurrieron.
    """
    tema = TEMAS[tema_nombre]
    figura = _figura(tema, alto=4.8)
    eje_onda, eje_f0 = figura.subplots(2, 1, sharex=True, height_ratios=(1.0, 1.6))

    _vestir_eje(eje_onda, tema)
    tiempo, minimos, maximos = _envolvente(analisis.amplitud)
    eje_onda.fill_between(tiempo, minimos, maximos, color=tema["audio"], linewidth=0.0, alpha=0.55)
    eje_onda.axhline(0.0, color=tema["rejilla"], linewidth=0.8)
    limite = max(float(np.max(np.abs(analisis.amplitud))) * 1.1, 0.01)
    eje_onda.set_ylim(-limite, limite)
    _etiquetar(eje_onda, tema, y="amplitud")
    _titular(eje_onda, tema, "Contorno de F0 — la entonación, junto a la forma de onda")

    _vestir_eje(eje_f0, tema)
    f0 = np.where(np.isfinite(analisis.f0), analisis.f0, np.nan)
    eje_f0.plot(analisis.tiempo_frames, f0, color=tema["audio"], linewidth=2.0, solid_capstyle="round")
    eje_f0.set_xlim(0.0, analisis.audio.duracion_s)
    _etiquetar(eje_f0, tema, x="tiempo (s)", y="F0 (Hz)")

    con_voz = analisis.f0[np.isfinite(analisis.f0)]

    # El eje se ajusta a la entonación que realmente hay, no al rango de búsqueda
    # completo: con 65-400 Hz fijos, una voz que se mueve entre 100 y 220 Hz queda
    # aplastada en el tercio inferior del panel y la forma del contorno se pierde.
    if con_voz.size:
        margen = max((float(con_voz.max()) - float(con_voz.min())) * 0.18, 12.0)
        eje_f0.set_ylim(
            max(config.F0_MINIMO_HZ * 0.9, float(con_voz.min()) - margen),
            min(config.F0_MAXIMO_HZ * 1.02, float(con_voz.max()) + margen),
        )
    else:
        eje_f0.set_ylim(config.F0_MINIMO_HZ * 0.9, config.F0_MAXIMO_HZ * 1.02)

    if con_voz.size:
        media = float(np.mean(con_voz))
        inferior, superior = eje_f0.get_ylim()
        eje_f0.axhline(media, color=tema["umbral"], linewidth=1.0, linestyle=(0, (5, 4)))
        eje_f0.text(
            analisis.audio.duracion_s * 0.995,
            media + (superior - inferior) * 0.02,
            f"media {media:.0f} Hz",
            color=tema["umbral"],
            fontsize=TAMANOS["nota"],
            fontfamily=FAMILIA,
            ha="right",
            va="bottom",
        )
    else:
        eje_f0.text(
            0.5,
            0.5,
            "No se detectó voz en el audio",
            transform=eje_f0.transAxes,
            color=tema["texto_suave"],
            fontsize=TAMANOS["etiqueta"],
            fontfamily=FAMILIA,
            ha="center",
            va="center",
        )
    return _a_png(figura)


FIGURAS = {
    "forma_de_onda": grafica_forma_de_onda,
    "espectrograma": grafica_espectrograma,
    "mel": grafica_mel,
    "f0": grafica_f0,
}
"""Las cuatro figuras de la etapa, por nombre."""


def renderizar(
    analisis: Analisis,
    temas: Iterable[str] = ("claro", "oscuro"),
    figuras: Iterable[str] | None = None,
) -> dict[str, dict[str, bytes]]:
    """
    Dibuja las figuras pedidas en los temas pedidos.

    Devuelve `{nombre_de_figura: {nombre_de_tema: bytes_png}}`.
    """
    nombres = tuple(figuras) if figuras else tuple(FIGURAS)
    return {
        nombre: {tema: FIGURAS[nombre](analisis, tema) for tema in temas}
        for nombre in nombres
    }


def como_data_uri(png: bytes) -> str:
    """Envuelve un PNG para incrustarlo directo en la página, sin guardarlo."""
    import base64

    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
