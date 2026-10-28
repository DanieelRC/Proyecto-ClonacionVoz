"""Etapa 7: reconocimiento del locutor con la salida del Perceiver.

La misma matriz [32, 1024] que condiciona al GPT-2 sirve de huella del hablante.
Aquí se compara con la de locutores conocidos mediante dos clasificadores:

- KNN: los 32 vectores se promedian en uno solo de 1024 por audio.
- GMM: los 32 vectores son 32 observaciones del mismo locutor; un GMM por
  locutor y gana el de mayor verosimilitud media.

Normalizaciones (todas se ajustan solo con los audios de entrenamiento de cada
pliegue, para que el audio de prueba no influya en ellas):

KNN
  1. Promedio de las 32 consultas -> [1024].
  2. Estandarización z por dimensión (media 0, desviación 1). Las 1024
     dimensiones del Perceiver tienen escalas muy distintas; sin esto unas pocas
     dimensiones grandes dominarían la distancia euclidiana.
  La distancia coseno ignora además la norma del vector: solo compara direcciones.

GMM
  1. Centrado por consulta: a cada una de las 32 consultas se le resta su media
     sobre todos los audios de entrenamiento. Cada consulta tiene una dirección
     propia, común a todas las personas; restarla deja lo que distingue a cada
     locutor (igual que la normalización por media cepstral en MFCC).
  2. Estandarización z por dimensión sobre todas las observaciones.
  3. PCA a 64 dimensiones. Con 4 audios por locutor hay 128 observaciones: no
     alcanzan para estimar gaussianas en 1024 dimensiones.
  4. Covarianza diagonal y regularización 1e-3 por la misma razón.

Locutor desconocido: con las predicciones de prueba de la validación cruzada se
fija el umbral que acepta al 95 % de los aciertos (distancia del vecino más
cercano en KNN; log-verosimilitud media en GMM). Lo que queda fuera es
«desconocido».
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .audio import AudioInvalido
from .identidad import ModeloNoDisponible

FORMATO_REPORTE = "clonvoz-reconocimiento-locutor"
ARCHIVO_MODELO = "reconocedor.joblib"
ARCHIVO_REPORTE = "reporte.json"
ARCHIVO_EMBEDDINGS = "embeddings.npz"
CARPETA_EXPORTACION = "reporte"
DESCONOCIDO = "Desconocido"
NOMBRES_METRICA = {"coseno": "cosine", "euclidiana": "euclidean"}


def carpeta_locutores() -> Path:
    """Dónde viven el modelo entrenado y su reporte (modelos/locutores o CLONVOZ_LOCUTORES)."""
    return Path(os.environ.get("CLONVOZ_LOCUTORES", "modelos/locutores"))


# ----------------------------------------------------------------- audio

def ruta_ffmpeg() -> str:
    """FFmpeg de CLONVOZ_FFMPEG, del PATH o, si está instalado, de imageio-ffmpeg."""
    elegido = os.environ.get("CLONVOZ_FFMPEG") or shutil.which("ffmpeg")
    if not elegido:
        try:
            import imageio_ffmpeg
            elegido = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            elegido = None
    if not elegido:
        raise AudioInvalido("No se encontró FFmpeg para leer MP3/M4A. Instálalo y agrégalo al PATH, "
                            "o indica su ruta en la variable CLONVOZ_FFMPEG.")
    return elegido


def leer_audio(ruta) -> np.ndarray:
    """Cualquier formato -> mono float32 a 22 050 Hz.

    WAV, FLAC y OGG los lee libsndfile, igual que la etapa 1 (misma mezcla a mono
    y mismo remuestreo de librosa). MP3, M4A y el resto los decodifica FFmpeg
    directamente a mono y 22 050 Hz.
    """
    import librosa
    import soundfile as sf

    ruta = Path(ruta)
    try:
        datos, sr = sf.read(ruta, dtype="float32", always_2d=True)
    except Exception:  # libsndfile no conoce el contenedor: lo intenta FFmpeg
        datos = None
    if datos is not None:
        muestras = datos.mean(axis=1)
        if sr != config.SR:
            muestras = librosa.resample(np.ascontiguousarray(muestras), orig_sr=sr, target_sr=config.SR)
        return np.ascontiguousarray(muestras, dtype=np.float32)

    orden = [ruta_ffmpeg(), "-nostdin", "-v", "error", "-i", str(ruta), "-vn",
             "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(config.SR), "-"]
    try:
        proceso = subprocess.run(orden, capture_output=True, check=False, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioInvalido(f"FFmpeg no pudo leer {ruta.name}: {exc}") from exc
    if proceso.returncode != 0:
        detalle = proceso.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioInvalido(f"FFmpeg no pudo leer {ruta.name}: {detalle[-1] if detalle else 'error desconocido'}")
    return np.frombuffer(proceso.stdout, dtype=np.float32).copy()


def validar_muestras(muestras, duracion_minima, duracion_maxima) -> str | None:
    """Motivo de rechazo, o None si el audio sirve. Límites propios de esta etapa."""
    duracion = len(muestras) / config.SR
    if duracion < duracion_minima:
        return f"dura {duracion:.1f} s; el mínimo es {duracion_minima:g} s"
    if duracion > duracion_maxima:
        return f"dura {duracion:.1f} s; el máximo es {duracion_maxima:g} s"
    if not np.isfinite(muestras).all():
        return "tiene muestras no finitas"
    if np.sqrt(np.mean(np.square(muestras, dtype=np.float64))) < config.RMS_MINIMO:
        return "está en silencio o casi en silencio"
    return None


# ------------------------------------------------------ conjunto de datos

@dataclass
class Muestra:
    """Un audio del conjunto, ya pasado por el Perceiver."""

    locutor: str
    archivo: str          # relativo a la carpeta del conjunto
    duracion_s: float
    vectores: np.ndarray  # [32, 1024]


@dataclass
class Conjunto:
    muestras: list[Muestra]
    rechazados: list[dict] = field(default_factory=list)
    carpeta: str = ""

    @property
    def locutores(self) -> list[str]:
        return sorted({m.locutor for m in self.muestras})


def listar_audios(carpeta) -> list[tuple[str, Path]]:
    """(locutor, ruta) de cada audio: una subcarpeta por locutor, su nombre es la etiqueta."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta {carpeta}.")
    audios = []
    for subcarpeta in sorted(p for p in carpeta.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for ruta in sorted(subcarpeta.rglob("*")):
            if ruta.is_file() and ruta.suffix.lower() in config.LOCUTOR_EXTENSIONES:
                audios.append((subcarpeta.name, ruta))
    return audios


def _huella(ruta: Path) -> str:
    """Identifica el contenido de un archivo para reutilizar su embedding."""
    estado = ruta.stat()
    return hashlib.sha1(f"{ruta.resolve()}|{estado.st_size}|{estado.st_mtime_ns}".encode()).hexdigest()


def extraer_conjunto(carpeta, codificador, *, duracion_minima=config.LOCUTOR_DURACION_MINIMA_S,
                     duracion_maxima=config.LOCUTOR_DURACION_MAXIMA_S, cache=None,
                     al_avanzar=None) -> Conjunto:
    """Lee cada audio, calcula su mel y lo pasa por el Perceiver.

    `cache` es un .npz de una ejecución anterior: los archivos que no cambiaron no
    se vuelven a procesar (el Perceiver es lo único lento de todo el entrenamiento).
    """
    from .caracteristicas import mel_espectrograma

    previos = {}
    if cache is not None and Path(cache).is_file():
        with np.load(cache, allow_pickle=False) as datos:
            previos = {str(h): (float(d), v) for h, d, v in zip(datos["huellas"], datos["duraciones"], datos["vectores"])}

    audios = listar_audios(carpeta)
    conjunto = Conjunto([], carpeta=str(carpeta))
    huellas = []
    for indice, (locutor, ruta) in enumerate(audios, 1):
        relativo = ruta.relative_to(carpeta).as_posix()
        if al_avanzar:
            al_avanzar(indice, len(audios), relativo)
        huella = _huella(ruta)
        if huella in previos:
            duracion, vectores = previos[huella]
        else:
            try:
                muestras = leer_audio(ruta)
            except AudioInvalido as exc:
                conjunto.rechazados.append({"archivo": relativo, "motivo": str(exc)})
                continue
            motivo = validar_muestras(muestras, duracion_minima, duracion_maxima)
            if motivo:
                conjunto.rechazados.append({"archivo": relativo, "motivo": motivo})
                continue
            duracion = len(muestras) / config.SR
            vectores = codificador.procesar_mel(mel_espectrograma(muestras))[0]
        if not duracion_minima <= duracion <= duracion_maxima:
            conjunto.rechazados.append({"archivo": relativo, "motivo": f"dura {duracion:.1f} s, fuera de los límites"})
            continue
        conjunto.muestras.append(Muestra(locutor, relativo, duracion, np.asarray(vectores, dtype=np.float32)))
        huellas.append(huella)

    # Un locutor con un solo audio no puede estar a la vez en entrenamiento y en prueba.
    por_locutor = {}
    for m in conjunto.muestras:
        por_locutor[m.locutor] = por_locutor.get(m.locutor, 0) + 1
    solos = {l for l, n in por_locutor.items() if n < 2}
    if solos:
        for m in conjunto.muestras:
            if m.locutor in solos:
                conjunto.rechazados.append({"archivo": m.archivo,
                                            "motivo": "su locutor tiene menos de 2 audios válidos"})
        huellas = [h for h, m in zip(huellas, conjunto.muestras) if m.locutor not in solos]
        conjunto.muestras = [m for m in conjunto.muestras if m.locutor not in solos]

    if cache is not None and conjunto.muestras:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, huellas=np.array(huellas),
                            duraciones=np.array([m.duracion_s for m in conjunto.muestras]),
                            vectores=np.stack([m.vectores for m in conjunto.muestras]))
    return conjunto


# ---------------------------------------------------------- clasificadores

@dataclass
class Prediccion:
    locutor: str               # el más probable, aunque no supere el umbral
    puntaje: float             # distancia (KNN, menor es mejor) o log-verosimilitud media (GMM)
    candidatos: list[tuple]    # [(locutor, puntaje)] del mejor al peor


class ModeloKNN:
    """K vecinos más cercanos sobre el promedio de las 32 consultas."""

    nombre = "KNN"
    mayor_es_mejor = False
    unidad = "distancia"

    def __init__(self, k=1, metrica="coseno"):
        self.k = k
        self.metrica = metrica

    @property
    def configuracion(self) -> dict:
        return {"k": self.k, "metrica": self.metrica, "pesos": "inverso de la distancia"}

    def describir(self) -> str:
        return f"KNN (k={self.k}, {self.metrica})"

    @staticmethod
    def rasgos(vectores) -> np.ndarray:
        """[N, 32, 1024] o [32, 1024] -> promedio de las consultas."""
        vectores = np.asarray(vectores, dtype=np.float64)
        return vectores.mean(axis=-2)

    def ajustar(self, vectores, etiquetas):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler

        x = self.rasgos(vectores)
        self.escalador = StandardScaler().fit(x)
        self.clasificador = KNeighborsClassifier(
            n_neighbors=min(self.k, len(x)), metric=NOMBRES_METRICA[self.metrica],
            weights="distance", algorithm="brute").fit(self.escalador.transform(x), etiquetas)
        self.etiquetas = np.asarray(etiquetas)
        return self

    def predecir(self, vectores) -> Prediccion:
        x = self.escalador.transform(self.rasgos(vectores)[None, :])
        distancias, indices = self.clasificador.kneighbors(x, n_neighbors=len(self.etiquetas))
        # Distancia al audio más cercano de cada locutor.
        cercania = {}
        for d, i in zip(distancias[0], indices[0]):
            cercania.setdefault(str(self.etiquetas[i]), float(d))
        votos = dict(zip(map(str, self.clasificador.classes_), self.clasificador.predict_proba(x)[0]))
        orden = sorted(cercania, key=lambda l: (-votos[l], cercania[l]))
        return Prediccion(orden[0], cercania[orden[0]], [(l, cercania[l]) for l in orden])


class ModeloGMM:
    """Un GMM por locutor; las 32 consultas de un audio son 32 observaciones."""

    nombre = "GMM"
    mayor_es_mejor = True
    unidad = "log-verosimilitud media"

    def __init__(self, componentes=config.LOCUTOR_GMM_COMPONENTES, dimensiones=config.LOCUTOR_GMM_DIMENSIONES,
                 semilla=0):
        self.componentes = componentes
        self.dimensiones = dimensiones
        self.semilla = semilla

    @property
    def configuracion(self) -> dict:
        return {"componentes": self.componentes, "covarianza": "diagonal", "pca": self.dimensiones,
                "regularizacion": 1e-3}

    def describir(self) -> str:
        return f"GMM ({self.componentes} componentes, diagonal, PCA {self.dimensiones})"

    def _observaciones(self, vectores) -> np.ndarray:
        """[..., 32, 1024] -> observaciones normalizadas [..., 32, d]."""
        vectores = np.asarray(vectores, dtype=np.float64) - self.media_consultas
        forma = vectores.shape
        planas = self.pca.transform(self.escalador.transform(vectores.reshape(-1, forma[-1])))
        return planas.reshape(*forma[:-1], -1)

    def ajustar(self, vectores, etiquetas):
        from sklearn.decomposition import PCA
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        vectores = np.asarray(vectores, dtype=np.float64)
        etiquetas = np.asarray(etiquetas)
        self.media_consultas = vectores.mean(axis=0)               # [32, 1024]
        centrados = (vectores - self.media_consultas).reshape(-1, vectores.shape[-1])
        self.escalador = StandardScaler().fit(centrados)
        dimensiones = min(self.dimensiones, centrados.shape[0] - 1, centrados.shape[1])
        self.pca = PCA(n_components=dimensiones, random_state=self.semilla).fit(self.escalador.transform(centrados))
        observaciones = self._observaciones(vectores)
        self.gmms = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for locutor in sorted(set(etiquetas.tolist())):
                propias = observaciones[etiquetas == locutor].reshape(-1, dimensiones)
                self.gmms[str(locutor)] = GaussianMixture(
                    n_components=min(self.componentes, len(propias)), covariance_type="diag",
                    reg_covar=1e-3, max_iter=200, n_init=1, random_state=self.semilla).fit(propias)
        return self

    def predecir(self, vectores) -> Prediccion:
        observaciones = self._observaciones(vectores)
        puntajes = {l: float(g.score(observaciones)) for l, g in self.gmms.items()}
        orden = sorted(puntajes, key=puntajes.get, reverse=True)
        return Prediccion(orden[0], puntajes[orden[0]], [(l, puntajes[l]) for l in orden])


def es_desconocido(modelo, puntaje, umbral) -> bool:
    if umbral is None:
        return False
    return puntaje < umbral if modelo.mayor_es_mejor else puntaje > umbral


# ---------------------------------------------------------- evaluación

def asignar_pliegues(conjunto, pliegues=config.LOCUTOR_PLIEGUES, semilla=0) -> np.ndarray:
    """Pliegue de prueba de cada muestra.

    Los audios de cada locutor se barajan con la semilla y el i-ésimo va al
    pliegue i mod P. Con 5 audios y 5 pliegues: 4 de entrenamiento y 1 de prueba
    por locutor en cada pliegue, y cada audio es de prueba exactamente una vez.
    """
    generador = np.random.default_rng(semilla)
    asignacion = np.zeros(len(conjunto.muestras), dtype=int)
    for locutor in conjunto.locutores:
        indices = [i for i, m in enumerate(conjunto.muestras) if m.locutor == locutor]
        indices.sort(key=lambda i: conjunto.muestras[i].archivo)
        for posicion, i in enumerate(generador.permutation(indices)):
            asignacion[i] = posicion % pliegues
    return asignacion


def validar_cruzado(fabrica, conjunto, asignacion) -> list[Prediccion | None]:
    """Predicción fuera de pliegue de cada muestra: nunca se entrena con el audio que se prueba."""
    vectores = np.stack([m.vectores for m in conjunto.muestras])
    etiquetas = np.array([m.locutor for m in conjunto.muestras])
    predicciones = [None] * len(etiquetas)
    for pliegue in sorted(set(asignacion.tolist())):
        prueba = asignacion == pliegue
        modelo = fabrica().ajustar(vectores[~prueba], etiquetas[~prueba])
        for i in np.flatnonzero(prueba):
            predicciones[i] = modelo.predecir(vectores[i])
    return predicciones


def metricas(etiquetas, predichos, locutores, asignacion) -> dict:
    """Exactitud global y por pliegue, recall por locutor y su promedio (macro), matriz de confusión."""
    etiquetas = np.asarray(etiquetas)
    predichos = np.asarray(predichos)
    indice = {l: i for i, l in enumerate(locutores)}
    matriz = np.zeros((len(locutores), len(locutores)), dtype=int)
    for real, predicho in zip(etiquetas, predichos):
        matriz[indice[real], indice[predicho]] += 1
    recall = {l: float(matriz[i, i] / matriz[i].sum()) for l, i in indice.items() if matriz[i].sum()}
    por_pliegue = [float(np.mean(etiquetas[asignacion == p] == predichos[asignacion == p]))
                   for p in sorted(set(asignacion.tolist()))]
    return {
        "exactitud": float(np.mean(etiquetas == predichos)),
        "recall_macro": float(np.mean(list(recall.values()))),
        "exactitud_pliegues": por_pliegue,
        "recall_por_locutor": recall,
        "matriz_confusion": matriz.tolist(),
    }


def calibrar_umbral(modelo, predicciones, etiquetas, aceptacion=config.LOCUTOR_ACEPTACION) -> dict:
    """Umbral que acepta la fracción `aceptacion` de los aciertos de la validación cruzada."""
    aciertos = np.array([p.puntaje for p, e in zip(predicciones, etiquetas) if p.locutor == e])
    errores = np.array([p.puntaje for p, e in zip(predicciones, etiquetas) if p.locutor != e])
    if not len(aciertos):
        return {"umbral": None, "aceptacion": aceptacion, "aciertos_rechazados": None, "errores_rechazados": None}
    percentil = (1 - aceptacion) * 100 if modelo.mayor_es_mejor else aceptacion * 100
    umbral = float(np.percentile(aciertos, percentil))
    rechazado = (lambda p: p < umbral) if modelo.mayor_es_mejor else (lambda p: p > umbral)
    return {
        "umbral": umbral,
        "aceptacion": aceptacion,
        "aciertos_rechazados": float(np.mean(rechazado(aciertos))),
        "errores_rechazados": float(np.mean(rechazado(errores))) if len(errores) else None,
    }


def _mejor(a, b) -> bool:
    """¿El resultado a supera a b? Primero exactitud, luego recall macro."""
    return (round(a["exactitud"], 6), round(a["recall_macro"], 6)) > (round(b["exactitud"], 6), round(b["recall_macro"], 6))


@dataclass
class Entrenamiento:
    reporte: dict
    reconocedor: "Reconocedor"


def entrenar(conjunto, *, pliegues=config.LOCUTOR_PLIEGUES, ks=config.LOCUTOR_K,
             metricas_knn=config.LOCUTOR_METRICAS, componentes=config.LOCUTOR_GMM_COMPONENTES,
             dimensiones=config.LOCUTOR_GMM_DIMENSIONES, aceptacion=config.LOCUTOR_ACEPTACION,
             semilla=0, limites=None, al_avanzar=None) -> Entrenamiento:
    """Validación cruzada de KNN (búsqueda de k y distancia) y GMM; reentrena el mejor con todo.

    1. Pliegues por locutor (asignar_pliegues).
    2. KNN: cada combinación de k y distancia por validación cruzada; se queda la mejor.
    3. GMM con componentes fijos, misma validación.
    4. Umbrales de desconocido con las predicciones de prueba.
    5. Se elige el de mayor exactitud (empate: recall macro; empate total: KNN, el más simple).
    6. Ambos modelos se reentrenan con todos los audios; el elegido es el que se usa.
    """
    if len(conjunto.locutores) < 2:
        raise ValueError("Se necesitan al menos 2 locutores con 2 audios válidos cada uno.")
    avisar = al_avanzar or (lambda texto: None)
    etiquetas = [m.locutor for m in conjunto.muestras]
    locutores = conjunto.locutores
    maximo = max(etiquetas.count(l) for l in locutores)
    pliegues = max(2, min(pliegues, maximo))
    asignacion = asignar_pliegues(conjunto, pliegues, semilla)

    busqueda = []
    mejor_knn = None
    for metrica in metricas_knn:
        for k in ks:
            avisar(f"Validación cruzada de KNN (k={k}, {metrica})")
            inicio = time.perf_counter()
            predicciones = validar_cruzado(lambda: ModeloKNN(k, metrica), conjunto, asignacion)
            resultado = metricas(etiquetas, [p.locutor for p in predicciones], locutores, asignacion)
            fila = {"k": k, "metrica": metrica, "exactitud": resultado["exactitud"],
                    "recall_macro": resultado["recall_macro"],
                    "ms": round((time.perf_counter() - inicio) * 1000, 1)}
            busqueda.append(fila)
            if mejor_knn is None or _mejor(resultado, mejor_knn[1]):
                mejor_knn = (ModeloKNN(k, metrica), resultado, predicciones)

    avisar(f"Validación cruzada de GMM ({componentes} componentes)")
    gmm = ModeloGMM(componentes, dimensiones, semilla)
    predicciones_gmm = validar_cruzado(lambda: ModeloGMM(componentes, dimensiones, semilla), conjunto, asignacion)
    resultado_gmm = metricas(etiquetas, [p.locutor for p in predicciones_gmm], locutores, asignacion)

    candidatos = {"knn": mejor_knn, "gmm": (gmm, resultado_gmm, predicciones_gmm)}
    resumen = {}
    for clave, (modelo, resultado, predicciones) in candidatos.items():
        resumen[clave] = {"nombre": modelo.nombre, "descripcion": modelo.describir(),
                          "configuracion": modelo.configuracion, "unidad": modelo.unidad,
                          "mayor_es_mejor": modelo.mayor_es_mejor, **resultado,
                          **calibrar_umbral(modelo, predicciones, etiquetas, aceptacion)}
    elegido = "gmm" if _mejor(resultado_gmm, mejor_knn[1]) else "knn"
    otro = "knn" if elegido == "gmm" else "gmm"
    motivo = (f"{resumen[elegido]['descripcion']}: exactitud {resumen[elegido]['exactitud']:.1%} y recall "
              f"{resumen[elegido]['recall_macro']:.1%}, frente a {resumen[otro]['exactitud']:.1%} y "
              f"{resumen[otro]['recall_macro']:.1%} del {resumen[otro]['nombre']}.")

    avisar("Reentrenando con todos los audios")
    vectores = np.stack([m.vectores for m in conjunto.muestras])
    finales = {clave: candidatos[clave][0].ajustar(vectores, etiquetas) for clave in candidatos}
    proyeccion = Proyeccion().ajustar(vectores, etiquetas)

    reporte = {
        "formato": FORMATO_REPORTE,
        "version": 1,
        "entrenado": datetime.now(timezone.utc).isoformat(),
        "carpeta": conjunto.carpeta,
        "locutores": locutores,
        "audios_por_locutor": {l: etiquetas.count(l) for l in locutores},
        "n_audios": len(etiquetas),
        "rechazados": conjunto.rechazados,
        "limites": limites or {},
        "validacion": {
            "pliegues": pliegues, "semilla": semilla,
            "esquema": (f"{pliegues} pliegues estratificados por locutor: en cada uno, un audio de cada locutor "
                        f"es de prueba y el resto de entrenamiento. Cada audio se prueba una sola vez."),
        },
        "normalizacion": {
            "knn": ["Promedio de las 32 consultas del Perceiver → 1 vector de 1024.",
                    "Estandarización z por dimensión, ajustada con los audios de entrenamiento.",
                    "Distancia coseno o euclidiana (búsqueda); votos ponderados por 1/distancia."],
            "gmm": ["Cada una de las 32 consultas es una observación del locutor.",
                    "Centrado por consulta: se resta la media de cada consulta en el entrenamiento.",
                    "Estandarización z por dimensión.",
                    f"PCA a {finales['gmm'].pca.n_components_} dimensiones.",
                    f"Un GMM por locutor: {componentes} componentes, covarianza diagonal, regularización 1e-3.",
                    "Gana el locutor con mayor log-verosimilitud media de las 32 observaciones."],
            "desconocido": (f"Umbral que acepta al {aceptacion:.0%} de los aciertos de la validación cruzada: "
                            "distancia al audio más cercano del locutor (KNN) o log-verosimilitud media (GMM)."),
        },
        "busqueda_knn": busqueda,
        "modelos": resumen,
        "elegido": elegido,
        "motivo_eleccion": motivo,
        "predicciones": [
            {"archivo": m.archivo, "locutor": m.locutor, "pliegue": int(p),
             "knn": a.locutor, "knn_puntaje": a.puntaje, "gmm": b.locutor, "gmm_puntaje": b.puntaje}
            for m, p, a, b in zip(conjunto.muestras, asignacion, mejor_knn[2], predicciones_gmm)],
        "proyeccion": proyeccion.como_dict(),
    }
    reconocedor = Reconocedor(locutores, finales, {c: resumen[c]["umbral"] for c in finales}, elegido, proyeccion,
                              {c: resumen[c]["descripcion"] for c in finales}, reporte["entrenado"])
    return Entrenamiento(reporte, reconocedor)


class Proyeccion:
    """PCA a 2 dimensiones del vector promedio estandarizado, para dibujar el espacio de locutores."""

    def ajustar(self, vectores, etiquetas):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        x = ModeloKNN.rasgos(vectores)
        self.escalador = StandardScaler().fit(x)
        self.pca = PCA(n_components=2, random_state=0).fit(self.escalador.transform(x))
        self.puntos = self.pca.transform(self.escalador.transform(x))
        self.etiquetas = list(etiquetas)
        return self

    def punto(self, vectores) -> list[float]:
        x = self.escalador.transform(ModeloKNN.rasgos(vectores)[None, :])
        return [float(v) for v in self.pca.transform(x)[0]]

    def como_dict(self) -> dict:
        return {"puntos": np.round(self.puntos, 4).tolist(), "etiquetas": self.etiquetas,
                "varianza": [float(v) for v in self.pca.explained_variance_ratio_]}


# ---------------------------------------------------------- reconocedor

@dataclass
class Identificacion:
    """Resultado de identificar un audio con el modelo elegido (y, a modo de contraste, el otro)."""

    locutor: str               # nombre o DESCONOCIDO
    modelo: str                # "knn" o "gmm"
    descripcion: str
    puntaje: float
    umbral: float | None
    mas_parecido: str          # el locutor más cercano, aunque no supere el umbral
    candidatos: list[dict]
    otro: dict
    punto: list[float]
    ms: float
    entrenado: str
    proyeccion: dict

    @property
    def desconocido(self) -> bool:
        return self.locutor == DESCONOCIDO

    def como_dict(self) -> dict:
        return {
            "locutor": self.locutor, "desconocido": self.desconocido, "modelo": self.modelo,
            "nombre_modelo": self.modelo.upper(), "descripcion": self.descripcion,
            "puntaje": self.puntaje, "umbral": self.umbral, "mas_parecido": self.mas_parecido,
            "candidatos": self.candidatos, "otro": self.otro, "punto": self.punto,
            "ms": round(self.ms, 2), "entrenado": self.entrenado,
        }


class Reconocedor:
    """Los dos modelos entrenados con todos los audios; identifica con el elegido."""

    def __init__(self, locutores, modelos, umbrales, elegido, proyeccion, descripciones, entrenado):
        self.locutores = locutores
        self.modelos = modelos
        self.umbrales = umbrales
        self.elegido = elegido
        self.proyeccion = proyeccion
        self.descripciones = descripciones
        self.entrenado = entrenado

    def _decidir(self, clave, vectores) -> dict:
        modelo = self.modelos[clave]
        prediccion = modelo.predecir(vectores)
        desconocido = es_desconocido(modelo, prediccion.puntaje, self.umbrales[clave])
        return {"modelo": clave, "nombre_modelo": clave.upper(), "descripcion": self.descripciones[clave],
                "locutor": DESCONOCIDO if desconocido else prediccion.locutor,
                "mas_parecido": prediccion.locutor, "puntaje": prediccion.puntaje,
                "umbral": self.umbrales[clave], "unidad": modelo.unidad,
                "mayor_es_mejor": modelo.mayor_es_mejor,
                "candidatos": [{"locutor": l, "puntaje": p} for l, p in prediccion.candidatos[:3]]}

    def identificar(self, vectores) -> Identificacion:
        vectores = np.asarray(vectores, dtype=np.float32)
        dimension = self.proyeccion.escalador.n_features_in_
        if vectores.shape != (32, dimension) or not np.isfinite(vectores).all():
            raise ValueError(f"Se esperaba la salida del Perceiver: una matriz finita [32, {dimension}].")
        inicio = time.perf_counter()
        principal = self._decidir(self.elegido, vectores)
        otro = self._decidir("gmm" if self.elegido == "knn" else "knn", vectores)
        punto = self.proyeccion.punto(vectores)
        return Identificacion(principal["locutor"], self.elegido, principal["descripcion"], principal["puntaje"],
                              principal["umbral"], principal["mas_parecido"], principal["candidatos"], otro,
                              punto, (time.perf_counter() - inicio) * 1000, self.entrenado,
                              self.proyeccion.como_dict())


# ---------------------------------------------------------- disco

def guardar(entrenamiento, carpeta):
    """Escribe el modelo (joblib), el reporte (JSON) y los archivos exportables."""
    import joblib

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    joblib.dump(entrenamiento.reconocedor, carpeta / ARCHIVO_MODELO)
    (carpeta / ARCHIVO_REPORTE).write_text(json.dumps(entrenamiento.reporte, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    from .graficas_locutores import exportar_reporte
    return exportar_reporte(entrenamiento.reporte, carpeta / CARPETA_EXPORTACION)


def cargar_reconocedor(carpeta=None) -> Reconocedor:
    """Lee el modelo entrenado. Solo se cargan archivos creados por cli_locutores en esta máquina."""
    carpeta = Path(carpeta or carpeta_locutores())
    ruta = carpeta / ARCHIVO_MODELO
    if not ruta.is_file():
        raise ModeloNoDisponible(f"No hay un reconocedor entrenado en {carpeta}. Entrénalo con "
                                 "python -m clonvoz.cli_locutores entrenar <carpeta de audios>.")
    try:
        import joblib
        reconocedor = joblib.load(ruta)
    except Exception as exc:
        raise ModeloNoDisponible(f"No se pudo leer {ruta}: {exc}. Vuelve a entrenar.") from exc
    if not isinstance(reconocedor, Reconocedor):
        raise ModeloNoDisponible(f"{ruta} no contiene un reconocedor de locutores.")
    return reconocedor


def cargar_reporte(carpeta=None) -> dict:
    ruta = Path(carpeta or carpeta_locutores()) / ARCHIVO_REPORTE
    if not ruta.is_file():
        raise ModeloNoDisponible("Todavía no hay un reporte de entrenamiento.")
    reporte = json.loads(ruta.read_text(encoding="utf-8"))
    if reporte.get("formato") != FORMATO_REPORTE:
        raise ModeloNoDisponible(f"{ruta} no es un reporte de reconocimiento de locutor.")
    return reporte
