"""Figuras y archivos exportables del reconocimiento del locutor (etapa 7)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

from .graficas import FAMILIA, TAMANOS, TEMAS, _etiquetar, _figura, _titular, _vestir_eje

# 40 colores distinguibles: alcanzan para los ~36 locutores del conjunto.
PALETA = list(colormaps["tab20"].colors) + list(colormaps["tab20b"].colors)


def _escala(tema) -> LinearSegmentedColormap:
    """Del fondo (cero audios) al color de la ruta de audio: las celdas vacías no compiten con la diagonal."""
    extremo = "#0b2a55" if tema["superficie"] == "#ffffff" else "#dbeafe"
    return LinearSegmentedColormap.from_list("conteo", [tema["superficie"], tema["audio"], extremo])


def _png(figura) -> bytes:
    archivo = io.BytesIO()
    figura.savefig(archivo, format="png", facecolor=figura.get_facecolor())
    return archivo.getvalue()


def matriz_confusion(resumen, locutores, tema_nombre) -> bytes:
    """Filas: locutor real. Columnas: locutor predicho. Cada celda cuenta audios de prueba."""
    tema = TEMAS[tema_nombre]
    matriz = np.asarray(resumen["matriz_confusion"])
    n = len(locutores)
    lado = min(13.0, max(6.0, 2.6 + 0.24 * n))
    figura = _figura(tema, alto=lado, ancho=lado + 0.8)
    eje = figura.add_subplot(111)
    _vestir_eje(eje, tema, con_rejilla=False)
    imagen = eje.imshow(matriz, cmap=_escala(tema), vmin=0, vmax=max(1, matriz.max()),
                        interpolation="nearest")
    eje.set_xticks(np.arange(-0.5, n), minor=True)
    eje.set_yticks(np.arange(-0.5, n), minor=True)
    eje.grid(True, which="minor", color=tema["rejilla"], linewidth=0.5)
    eje.tick_params(which="minor", length=0)
    fuente = max(8.0, min(TAMANOS["tick"], 330 / max(n, 1)))
    eje.set_xticks(range(n), locutores, rotation=90, fontsize=fuente)
    eje.set_yticks(range(n), locutores, fontsize=fuente)
    limite = matriz.max() / 2 if matriz.max() else 0.5
    for i, j in zip(*np.nonzero(matriz)):
        # Celdas fuertes: texto blanco en claro (fondo oscuro) y oscuro en el tema oscuro (fondo claro).
        fuerte = matriz[i, j] > limite
        if tema_nombre == "claro":
            color = "#ffffff" if fuerte else tema["texto"]
        else:
            color = "#0f172a" if fuerte else tema["texto"]
        eje.text(j, i, str(matriz[i, j]), ha="center", va="center", fontsize=fuente,
                 color=color, fontfamily=FAMILIA)
    _titular(eje, tema, f"{resumen['descripcion']} · exactitud {resumen['exactitud']:.1%} · "
                        f"recall {resumen['recall_macro']:.1%}", ancho=lado)
    _etiquetar(eje, tema, x="Locutor predicho", y="Locutor real")
    barra = figura.colorbar(imagen, ax=eje, fraction=0.04, pad=0.02)
    barra.ax.tick_params(colors=tema["texto_suave"], labelsize=TAMANOS["tick"])
    barra.set_label("Audios de prueba", color=tema["texto_suave"], fontsize=TAMANOS["nota"])
    return _png(figura)


def proyeccion(datos, tema_nombre, punto=None, etiqueta=None) -> bytes:
    """PCA 2D de los audios de entrenamiento; el audio actual como estrella."""
    tema = TEMAS[tema_nombre]
    figura = _figura(tema, alto=6.4, ancho=9.0)
    eje = figura.add_subplot(111)
    _vestir_eje(eje, tema)
    eje.grid(True, axis="x", color=tema["rejilla"], linewidth=0.8)
    puntos = np.asarray(datos["puntos"])
    etiquetas = np.asarray(datos["etiquetas"])
    for indice, locutor in enumerate(sorted(set(etiquetas.tolist()))):
        propios = puntos[etiquetas == locutor]
        color = PALETA[indice % len(PALETA)]
        eje.scatter(propios[:, 0], propios[:, 1], s=34, color=color, alpha=0.85, linewidths=0)
        centro = propios.mean(axis=0)
        eje.annotate(locutor, centro, fontsize=TAMANOS["nota"] - 1.5, color=tema["texto_suave"],
                     ha="center", va="bottom", xytext=(0, 5), textcoords="offset points", fontfamily=FAMILIA)
    if punto is not None:
        eje.scatter([punto[0]], [punto[1]], s=320, marker="*", color=tema["audio"],
                    edgecolors=tema["texto"], linewidths=1.0, zorder=5)
        if etiqueta:
            eje.annotate(etiqueta, punto, fontsize=TAMANOS["etiqueta"], color=tema["texto"], zorder=6,
                         xytext=(10, -14), textcoords="offset points", fontfamily=FAMILIA)
    varianza = datos.get("varianza") or [0, 0]
    _titular(eje, tema, "Espacio de locutores: promedio del Perceiver proyectado con PCA", ancho=9.0)
    _etiquetar(eje, tema, x=f"Componente 1 ({varianza[0]:.0%} de la varianza)",
               y=f"Componente 2 ({varianza[1]:.0%})")
    return _png(figura)


def renderizar_reporte(reporte) -> dict:
    """{figura: {tema: png}} para la página: las dos matrices y la proyección."""
    figuras = {}
    for clave, resumen in reporte["modelos"].items():
        figuras[f"matriz_{clave}"] = {t: matriz_confusion(resumen, reporte["locutores"], t) for t in TEMAS}
    figuras["proyeccion"] = {t: proyeccion(reporte["proyeccion"], t) for t in TEMAS}
    return figuras


def renderizar_identificacion(identificacion) -> dict:
    etiqueta = "Este audio → " + identificacion.locutor
    return {"proyeccion": {t: proyeccion(identificacion.proyeccion, t, identificacion.punto, etiqueta)
                           for t in TEMAS}}


def _csv(ruta, encabezado, filas):
    # utf-8-sig: Excel en Windows abre bien los acentos.
    with open(ruta, "w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezado)
        escritor.writerows(filas)


def exportar_reporte(reporte, carpeta) -> list[Path]:
    """Escribe en `carpeta` las figuras (PNG, tema claro) y las tablas (CSV) del entrenamiento."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    locutores = reporte["locutores"]
    modelos = reporte["modelos"]
    escritos = []

    for clave, resumen in modelos.items():
        ruta = carpeta / f"matriz_confusion_{clave}.png"
        ruta.write_bytes(matriz_confusion(resumen, locutores, "claro"))
        escritos.append(ruta)
        ruta = carpeta / f"matriz_confusion_{clave}.csv"
        _csv(ruta, ["real \\ predicho", *locutores],
             [[real, *fila] for real, fila in zip(locutores, resumen["matriz_confusion"])])
        escritos.append(ruta)

    ruta = carpeta / "proyeccion_pca.png"
    ruta.write_bytes(proyeccion(reporte["proyeccion"], "claro"))
    escritos.append(ruta)

    ruta = carpeta / "metricas.csv"
    _csv(ruta, ["modelo", "configuracion", "exactitud", "recall_macro", "exactitud_por_pliegue",
                "umbral_desconocido", "aciertos_rechazados", "errores_rechazados", "en_uso"],
         [[r["nombre"], r["descripcion"], f"{r['exactitud']:.4f}", f"{r['recall_macro']:.4f}",
           " ".join(f"{v:.4f}" for v in r["exactitud_pliegues"]),
           "" if r["umbral"] is None else f"{r['umbral']:.6f}",
           "" if r["aciertos_rechazados"] is None else f"{r['aciertos_rechazados']:.4f}",
           "" if r["errores_rechazados"] is None else f"{r['errores_rechazados']:.4f}",
           "sí" if clave == reporte["elegido"] else "no"]
          for clave, r in modelos.items()])
    escritos.append(ruta)

    ruta = carpeta / "busqueda_knn.csv"
    _csv(ruta, ["k", "distancia", "exactitud", "recall_macro"],
         [[f["k"], f["metrica"], f"{f['exactitud']:.4f}", f"{f['recall_macro']:.4f}"]
          for f in reporte["busqueda_knn"]])
    escritos.append(ruta)

    ruta = carpeta / "recall_por_locutor.csv"
    _csv(ruta, ["locutor", "audios", "recall_knn", "recall_gmm"],
         [[l, reporte["audios_por_locutor"][l],
           f"{modelos['knn']['recall_por_locutor'].get(l, 0):.4f}",
           f"{modelos['gmm']['recall_por_locutor'].get(l, 0):.4f}"] for l in locutores])
    escritos.append(ruta)

    ruta = carpeta / "predicciones.csv"
    _csv(ruta, ["archivo", "locutor_real", "pliegue", "knn", "knn_distancia", "gmm", "gmm_log_verosimilitud"],
         [[p["archivo"], p["locutor"], p["pliegue"] + 1, p["knn"], f"{p['knn_puntaje']:.6f}",
           p["gmm"], f"{p['gmm_puntaje']:.4f}"] for p in reporte["predicciones"]])
    escritos.append(ruta)

    if reporte["rechazados"]:
        ruta = carpeta / "rechazados.csv"
        _csv(ruta, ["archivo", "motivo"], [[r["archivo"], r["motivo"]] for r in reporte["rechazados"]])
        escritos.append(ruta)

    ruta = carpeta / "reporte.json"
    ruta.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    escritos.append(ruta)
    return escritos
