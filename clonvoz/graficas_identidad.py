"""Vistas de diagnóstico: transforman colores, nunca los vectores del modelo."""
import io
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import PowerNorm
from matplotlib import colormaps
from .graficas import TEMAS, FAMILIA, TAMANOS
from . import config

PIXELES_POR_FRAME = 2
"""Ancho que se le da a cada instante del mel en los mapas de atención."""

ANCHO_MAXIMO_ATENCION = 1600
"""
Tope de ancho del mapa de atención, en píxeles.

Sin tope, una referencia de 30 s (T = 2 584) producía un PNG de 5 388 px de
ancho: cuatro de esos por referencia, ocho al comparar dos, todos en base64
dentro de la misma respuesta, y luego mostrados en una tarjeta de unos 600 px.
RNF-02 pide justamente lo contrario, imágenes en baja resolución.
"""


def similitud_consultas(vectores):
    """Coseno entre filas. Una fila nula no tiene coseno: se muestra sin dato."""
    valores = np.asarray(vectores, dtype=np.float64)
    normas = np.linalg.norm(valores, axis=1)
    denominador = normas[:, None] * normas[None, :]
    salida = np.full((len(valores), len(valores)), np.nan)
    np.divide(valores @ valores.T, denominador, out=salida, where=denominador > 0)
    return np.clip(salida, -1, 1)


def agrupar_columnas(matriz, columnas_maximas):
    """Reduce [32, T] a [32, <= columnas_maximas] promediando frames vecinos.

    Devuelve `(matriz, frames_por_columna)`. Se promedia y no se toma el máximo
    a propósito: el promedio conserva la proporción real del peso en ese tramo
    de tiempo, mientras que el máximo la exageraría, y en esta etapa los pesos
    se muestran sin renormalizar ni inflar.

    La matriz original no se toca: la agrupación afecta solo a la imagen. Las
    cifras del pie y los .npy que exporta la CLI siguen saliendo del tensor
    completo.
    """
    frames = matriz.shape[1]
    if frames <= columnas_maximas:
        return matriz, 1
    # Cuántos frames entran en cada columna dibujada.
    por_columna = int(np.ceil(frames / columnas_maximas))
    # La división entera deja un resto que no llena una columna; se promedia
    # aparte para no perder el final del audio.
    utiles = (frames // por_columna) * por_columna
    agrupada = matriz[:, :utiles].reshape(matriz.shape[0], -1, por_columna).mean(axis=2)
    resto = matriz[:, utiles:]
    if resto.shape[1]:
        agrupada = np.concatenate((agrupada, resto.mean(axis=1, keepdims=True)), axis=1)
    return agrupada, por_columna


def renderizar_identidades(resultados):
    """1. Preparar vistas. 2. Compartir escalas. 3. Dibujar sin alterar datos.

    Devuelve, por resultado, `{vista: {tema: bytes_png}}`. Entrega los bytes
    crudos y no un "data:image/png;base64,...": envolverlos para la web es
    trabajo de quien sirve la página, igual que en `graficas.py`. Así la CLI
    escribe el PNG directamente en vez de deshacer una codificación.

    La potencia 0.4 amplía el contraste de pesos pequeños, manteniendo el cero.
    La barra indica pesos originales, no porcentajes ni probabilidades nuevas.
    Si no existe atención positiva se muestra un mensaje, nunca color inventado.
    """
    vistas = []
    for r in resultados:
        completa = r.capas_atencion[0][:, 32:] if r.capas_atencion else r.atencion
        # Los mapas de atención se agrupan para dibujarlos; las cifras del pie
        # se siguen calculando sobre la matriz completa.
        ultima, agrupacion = agrupar_columnas(r.atencion, ANCHO_MAXIMO_ATENCION // PIXELES_POR_FRAME)
        inicial, _ = agrupar_columnas(completa, ANCHO_MAXIMO_ATENCION // PIXELES_POR_FRAME)
        vistas.append({
            'mapas': {
                'atencion': ultima,
                'atencion_inicial': inicial,
                'similitud': similitud_consultas(r.vectores),
                'centrada': r.vectores - r.vectores.mean(axis=0, keepdims=True),
                'identidad': r.vectores,
            },
            # Sin agrupar: de aquí salen "peso medio" y "máximo" del pie.
            'originales': {'atencion': r.atencion, 'atencion_inicial': completa},
            'agrupacion': agrupacion,
        })
    titulos = {'atencion': 'Atención al audio · última capa',
               'atencion_inicial': 'Atención al audio · primera capa',
               'similitud': 'Similitud entre consultas',
               'centrada': 'Diferencia respecto a la consulta media',
               'identidad': 'Vectores originales de voz y estilo'}
    # Cada vista usa el mismo límite para las referencias que se comparan.
    limites = {nombre: max(float(np.nanmax(np.abs(v['mapas'][nombre]))) for v in vistas)
               for nombre in ('atencion', 'atencion_inicial', 'centrada', 'identidad')}
    imagenes = []
    for r, vista in zip(resultados, vistas):
        matrices = vista['mapas']
        figuras = {nombre: {} for nombre in matrices}
        for tema_nombre, tema in TEMAS.items():
            for nombre, matriz in matrices.items():
                es_atencion = nombre.startswith('atencion')
                # Dos píxeles por instante evitan descartar columnas estrechas.
                # Los 220 píxeles extra son márgenes y barra, no datos del mapa.
                ancho_mapa = max(660, PIXELES_POR_FRAME * matriz.shape[1]) if es_atencion else 660
                ancho = ancho_mapa + 220 if es_atencion else 880
                figura = Figure(figsize=(ancho / 110, 4.8), dpi=110, facecolor=tema['superficie'])
                FigureCanvasAgg(figura)
                # Posición fija: la barra de colores no debe reducir el mapa.
                eje = (figura.add_axes([80/ancho, .23, ancho_mapa/ancho, .67])
                       if es_atencion else figura.add_subplot(111))
                eje.set_facecolor(tema['superficie'])
                # Tamaños y alineación tomados de graficas.py para que estas
                # figuras se lean igual que las de la etapa 1 proyectadas (RNF-04, RNF-08).
                eje.set_title(titulos[nombre], loc='left', color=tema['texto'],
                              fontfamily=FAMILIA, fontsize=TAMANOS['titulo'])
                if es_atencion and not np.any(matriz > 0):
                    # Cero es un resultado numérico, no una región oscura útil.
                    eje.text(.5, .5, 'Todos los pesos al audio son cero en esta capa.\n'
                             'Esto no demuestra que el audio no se usó en otra capa.',
                             ha='center', va='center', transform=eje.transAxes,
                             color=tema['texto'], fontsize=TAMANOS['etiqueta'], wrap=True)
                    eje.set_axis_off()
                else:
                    colores = colormaps['magma' if es_atencion else 'RdBu_r'].copy()
                    colores.set_bad('#808080')
                    opciones = {'cmap': colores}
                    if es_atencion:
                        # No imponemos 1e-9: se utiliza el máximo realmente medido.
                        opciones['norm'] = PowerNorm(gamma=.4, vmin=0, vmax=limites[nombre])
                    else:
                        limite = 1 if nombre == 'similitud' else (limites[nombre] or 1)
                        opciones.update(vmin=-limite, vmax=limite)
                    imagen = eje.imshow(matriz, origin='lower', aspect='auto',
                                        interpolation='nearest', **opciones)
                    eje.set_ylabel('Consulta (1–32)', color=tema['texto'], fontsize=TAMANOS['etiqueta'])
                    eje.set_yticks([0, 7, 15, 23, 31], [1, 8, 16, 24, 32])
                    etiqueta = 'Dimensión (0–1023)'
                    if es_atencion:
                        etiqueta = 'Tiempo de referencia (s)'
                        posiciones = np.linspace(0, matriz.shape[1]-1, 5)
                        # Cada columna dibujada cubre `agrupacion` frames, así que
                        # el segundo de la columna p es p * agrupacion * salto / sr.
                        segundos = posiciones * vista['agrupacion'] * config.HOP_LENGTH / config.SR
                        eje.set_xticks(posiciones, [f'{s:.1f}' for s in segundos])
                    elif nombre == 'similitud':
                        etiqueta = 'Consulta (1–32)'
                        eje.set_xticks([0, 7, 15, 23, 31], [1, 8, 16, 24, 32])
                    eje.set_xlabel(etiqueta, color=tema['texto'], fontsize=TAMANOS['etiqueta'])
                    eje.tick_params(colors=tema['texto_suave'], labelsize=TAMANOS['tick'])
                    if es_atencion:
                        espacio_barra = figura.add_axes([(100 + ancho_mapa)/ancho, .23, 20/ancho, .67])
                        barra = figura.colorbar(imagen, cax=espacio_barra)
                    else:
                        barra = figura.colorbar(imagen, ax=eje, pad=.02)
                    barra.ax.tick_params(colors=tema['texto_suave'], labelsize=TAMANOS['tick'])
                    barra.set_label('Peso real · color: potencia 0.4' if es_atencion else
                                    ('Coseno (no identidad de personas)' if nombre == 'similitud' else 'Valor'),
                                    color=tema['texto_suave'], fontsize=TAMANOS['nota'])
                if es_atencion:
                    indice = 0 if nombre == 'atencion_inicial' else -1
                    propias = r.capas_atencion[indice][:, :32] if r.capas_atencion else r.atencion_consultas[0]
                    # Las cifras salen del tensor completo, no del agrupado.
                    original = vista['originales'][nombre]
                    nota_agrupacion = ''
                    if vista['agrupacion'] > 1:
                        milisegundos = vista['agrupacion'] * config.HOP_LENGTH / config.SR * 1000
                        nota_agrupacion = (f'\nColumnas agrupadas para dibujar: promedio de '
                                           f'{vista["agrupacion"]} frames ({milisegundos:.0f} ms)')
                    figura.text(.5, .025,
                                f'Peso medio al audio: {original.sum(axis=1).mean():.3e}  |  '
                                f'a consultas: {propias.sum(axis=1).mean():.3e}\n'
                                f'Máximo al audio: {original.max():.3e} · promedio de 8 cabezas'
                                f'{nota_agrupacion}',
                                ha='center', color=tema['texto_suave'], fontsize=TAMANOS['nota'])
                if not es_atencion:
                    figura.tight_layout()
                archivo = io.BytesIO()
                figura.savefig(archivo, format='png', facecolor=tema['superficie'])
                figuras[nombre][tema_nombre] = archivo.getvalue()
        imagenes.append(figuras)
    return imagenes
