"""Generación de mapas de atención y similitud de identidad para XTTS-v2."""

import io
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import PowerNorm
from matplotlib import colormaps
from .graficas import TEMAS, FAMILIA, TAMANOS
from . import config

PIXELES_POR_FRAME = 2
ANCHO_MAXIMO_ATENCION = 1600


def similitud_consultas(vectores):
    """Calcula la matriz de similitud coseno entre los vectores de consulta."""
    valores = np.asarray(vectores, dtype=np.float64)
    normas = np.linalg.norm(valores, axis=1)
    denominador = normas[:, None] * normas[None, :]
    salida = np.full((len(valores), len(valores)), np.nan)
    np.divide(valores @ valores.T, denominador, out=salida, where=denominador > 0)
    return np.clip(salida, -1, 1)


def agrupar_columnas(matriz, columnas_maximas):
    """Reduce el número de columnas promediando frames adyacentes para visualización."""
    frames = matriz.shape[1]
    if frames <= columnas_maximas:
        return matriz, 1

    por_columna = int(np.ceil(frames / columnas_maximas))
    utiles = (frames // por_columna) * por_columna
    agrupada = matriz[:, :utiles].reshape(matriz.shape[0], -1, por_columna).mean(axis=2)
    resto = matriz[:, utiles:]
    if resto.shape[1]:
        agrupada = np.concatenate((agrupada, resto.mean(axis=1, keepdims=True)), axis=1)
    return agrupada, por_columna


def renderizar_identidades(resultados):
    """Genera las figuras de atención, similitud y vectores para cada resultado en temas claro y oscuro."""
    vistas = []
    for r in resultados:
        completa = r.capas_atencion[0][:, 32:] if r.capas_atencion else r.atencion
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
            'originales': {'atencion': r.atencion, 'atencion_inicial': completa},
            'agrupacion': agrupacion,
        })

    titulos = {
        'atencion': 'Atención al audio · última capa',
        'atencion_inicial': 'Atención al audio · primera capa',
        'similitud': 'Similitud entre consultas',
        'centrada': 'Diferencia respecto a la consulta media',
        'identidad': 'Vectores originales de voz y estilo',
    }

    limites = {nombre: max(float(np.nanmax(np.abs(v['mapas'][nombre]))) for v in vistas)
               for nombre in ('atencion', 'atencion_inicial', 'centrada', 'identidad')}

    imagenes = []
    for r, vista in zip(resultados, vistas):
        matrices = vista['mapas']
        figuras = {nombre: {} for nombre in matrices}
        for tema_nombre, tema in TEMAS.items():
            for nombre, matriz in matrices.items():
                es_atencion = nombre.startswith('atencion')
                ancho_mapa = max(660, PIXELES_POR_FRAME * matriz.shape[1]) if es_atencion else 660
                ancho = ancho_mapa + 220 if es_atencion else 880
                figura = Figure(figsize=(ancho / 110, 4.8), dpi=110, facecolor=tema['superficie'])
                FigureCanvasAgg(figura)

                eje = (figura.add_axes([80/ancho, .23, ancho_mapa/ancho, .67])
                       if es_atencion else figura.add_subplot(111))
                eje.set_facecolor(tema['superficie'])
                eje.set_title(titulos[nombre], loc='left', color=tema['texto'],
                              fontfamily=FAMILIA, fontsize=TAMANOS['titulo'])

                if es_atencion and not np.any(matriz > 0):
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
