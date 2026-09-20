"""Vistas de diagnóstico: transforman colores, nunca los vectores del modelo."""
import io
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import PowerNorm
from matplotlib import colormaps
from .graficas import TEMAS, FAMILIA, como_data_uri
from . import config


def similitud_consultas(vectores):
    """Coseno entre filas. Una fila nula no tiene coseno: se muestra sin dato."""
    valores = np.asarray(vectores, dtype=np.float64)
    normas = np.linalg.norm(valores, axis=1)
    denominador = normas[:, None] * normas[None, :]
    salida = np.full((len(valores), len(valores)), np.nan)
    np.divide(valores @ valores.T, denominador, out=salida, where=denominador > 0)
    return np.clip(salida, -1, 1)


def renderizar_identidades(resultados):
    """1. Preparar vistas. 2. Compartir escalas. 3. Dibujar sin alterar datos.

    La potencia 0.4 amplía el contraste de pesos pequeños, manteniendo el cero.
    La barra indica pesos originales, no porcentajes ni probabilidades nuevas.
    Si no existe atención positiva se muestra un mensaje, nunca color inventado.
    """
    vistas = []
    for r in resultados:
        vistas.append({
            'atencion': r.atencion,
            'atencion_inicial': r.capas_atencion[0][:, 32:] if r.capas_atencion else r.atencion,
            'similitud': similitud_consultas(r.vectores),
            'centrada': r.vectores - r.vectores.mean(axis=0, keepdims=True),
            'identidad': r.vectores,
        })
    titulos = {'atencion': 'Atención al audio · última capa',
               'atencion_inicial': 'Atención al audio · primera capa',
               'similitud': 'Similitud entre consultas',
               'centrada': 'Diferencia respecto a la consulta media',
               'identidad': 'Vectores originales de voz y estilo'}
    # Cada vista usa el mismo límite para las referencias que se comparan.
    limites = {nombre: max(float(np.nanmax(np.abs(v[nombre]))) for v in vistas)
               for nombre in ('atencion', 'atencion_inicial', 'centrada', 'identidad')}
    imagenes = []
    for r, matrices in zip(resultados, vistas):
        figuras = {nombre: {} for nombre in matrices}
        for tema_nombre, tema in TEMAS.items():
            for nombre, matriz in matrices.items():
                es_atencion = nombre.startswith('atencion')
                # Dos píxeles por instante evitan descartar columnas estrechas.
                # Los 220 píxeles extra son márgenes y barra, no datos del mapa.
                ancho_mapa = max(660, 2 * matriz.shape[1]) if es_atencion else 660
                ancho = ancho_mapa + 220 if es_atencion else 880
                figura = Figure(figsize=(ancho / 110, 4.8), dpi=110, facecolor=tema['superficie'])
                FigureCanvasAgg(figura)
                # Posición fija: la barra de colores no debe reducir el mapa.
                eje = (figura.add_axes([80/ancho, .23, ancho_mapa/ancho, .67])
                       if es_atencion else figura.add_subplot(111))
                eje.set_facecolor(tema['superficie'])
                eje.set_title(titulos[nombre], color=tema['texto'], fontfamily=FAMILIA)
                if es_atencion and not np.any(matriz > 0):
                    # Cero es un resultado numérico, no una región oscura útil.
                    eje.text(.5, .5, 'Todos los pesos al audio son cero en esta capa.\n'
                             'Esto no demuestra que el audio no se usó en otra capa.',
                             ha='center', va='center', transform=eje.transAxes,
                             color=tema['texto'], wrap=True)
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
                    eje.set_ylabel('Consulta (1–32)', color=tema['texto'])
                    eje.set_yticks([0, 7, 15, 23, 31], [1, 8, 16, 24, 32])
                    etiqueta = 'Dimensión (0–1023)'
                    if es_atencion:
                        etiqueta = 'Tiempo de referencia (s)'
                        posiciones = np.linspace(0, matriz.shape[1]-1, 5)
                        eje.set_xticks(posiciones, [f'{p * config.HOP_LENGTH / config.SR:.1f}' for p in posiciones])
                    elif nombre == 'similitud':
                        etiqueta = 'Consulta (1–32)'
                        eje.set_xticks([0, 7, 15, 23, 31], [1, 8, 16, 24, 32])
                    eje.set_xlabel(etiqueta, color=tema['texto'])
                    eje.tick_params(colors=tema['texto_suave'])
                    if es_atencion:
                        espacio_barra = figura.add_axes([(100 + ancho_mapa)/ancho, .23, 20/ancho, .67])
                        barra = figura.colorbar(imagen, cax=espacio_barra)
                    else:
                        barra = figura.colorbar(imagen, ax=eje, pad=.02)
                    barra.ax.tick_params(colors=tema['texto_suave'])
                    barra.set_label('Peso real · color: potencia 0.4' if es_atencion else
                                    ('Coseno (no identidad de personas)' if nombre == 'similitud' else 'Valor'),
                                    color=tema['texto_suave'])
                if es_atencion:
                    indice = 0 if nombre == 'atencion_inicial' else -1
                    propias = r.capas_atencion[indice][:, :32] if r.capas_atencion else r.atencion_consultas[0]
                    figura.text(.5, .025,
                                f'Peso medio al audio: {matriz.sum(axis=1).mean():.3e}  |  '
                                f'a consultas: {propias.sum(axis=1).mean():.3e}\n'
                                f'Máximo al audio: {matriz.max():.3e} · promedio de 8 cabezas',
                                ha='center', color=tema['texto_suave'], fontsize=9)
                if not es_atencion:
                    figura.tight_layout()
                archivo = io.BytesIO()
                figura.savefig(archivo, format='png', facecolor=tema['superficie'])
                figuras[nombre][tema_nombre] = como_data_uri(archivo.getvalue())
        imagenes.append(figuras)
    return imagenes
