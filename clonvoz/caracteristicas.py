"""
Las cuatro características visuales de la etapa 1.

Procesamiento de señales puro: estas funciones reciben y devuelven arreglos de
numpy, y no saben nada de matplotlib ni de Flask. Son las que la etapa 2 va a
importar directamente para alimentar al Perceiver Resampler, así que se mantienen
libres de cualquier dependencia de presentación.

    forma_de_onda        las muestras tal cual, con su eje de tiempo
    espectrograma_stft   [1 + N_FFT/2, T] en dB          (librosa)
    mel_espectrograma    [80, T] log-mel de XTTS-v2      (torchaudio)
    contorno_f0          [T] en Hz, con detección de voz (librosa)

CÓMO LEER ESTE ARCHIVO
Cada función de cálculo lleva en su docstring una receta numerada, y esos números
aparecen otra vez como comentarios dentro del cuerpo: el paso 3 de la receta es la
línea marcada con "# 3.". Los comentarios anotan además en qué se convierte el
arreglo en cada paso, con la forma entre corchetes, porque ese recorrido —de [n]
muestras sueltas a un tensor [80, T]— es justamente lo que hay que poder explicar.

Un recordatorio de vocabulario que se usa en todo el archivo:

    n       número de muestras del audio (8 segundos a 22 050 Hz son 176 400)
    T       número de frames o ventanas de análisis; T = 1 + n // 256
    frame   una ventana de 1024 muestras (46 ms) sobre la que se mide algo
    salto   las 256 muestras (11.6 ms) que avanza una ventana respecto a la anterior
"""

from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np

from . import config


def forma_de_onda(muestras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve `(tiempo, amplitud)` de la señal.

    No hay cálculo que hacer: la forma de onda *son* las muestras. Se acompaña
    del eje de tiempo en segundos para que la gráfica no se dibuje en índices.

    La receta, paso a paso:
      1. Asegurar que las muestras sean números decimales de 32 bits.
      2. Construir el eje de tiempo: la muestra número i se grabó en el segundo
         i / 22050, porque eso es lo que significa "22 050 muestras por segundo".
    """
    # 1. [n] -> [n]. Solo fija el tipo; el contenido no se toca.
    amplitud = np.asarray(muestras, dtype=np.float32)

    # 2. [n] -> [n]. np.arange genera 0, 1, 2, … n-1 (el número de cada muestra) y
    #    al dividir entre la frecuencia de muestreo queda el segundo en que ocurrió
    #    cada una. Así el eje horizontal se etiqueta en segundos y no en índices.
    tiempo = np.arange(amplitud.shape[0], dtype=np.float32) / config.SR

    return tiempo, amplitud


def espectrograma_stft(muestras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Espectrograma por Transformada de Fourier de Tiempo Corto, en decibelios.

    Devuelve `(espectrograma_db, frecuencias_hz)`, con el espectrograma de forma
    [1 + N_FFT/2, T]. Se usa el mismo N_FFT y el mismo salto que el mel, así que
    ambos mapas de calor comparten eje temporal y se pueden comparar de frente,
    que es justo lo que pide el documento de etapas.

    La idea de fondo: la Transformada de Fourier dice qué frecuencias componen una
    señal, pero no cuándo suena cada una. "De tiempo corto" significa aplicarla no
    al audio entero sino a ventanas pequeñas que van avanzando, de modo que cada
    ventana produce una columna del mapa y así se recupera el eje del tiempo.

    La receta, paso a paso:
      1. Cortar la señal en ventanas de 1024 muestras que avanzan de 256 en 256, y
         aplicarle a cada una la Transformada de Fourier.
      2. Quedarse con la magnitud de cada número complejo: cuánta energía hay en esa
         frecuencia, descartando la fase.
      3. Pasar esa magnitud a decibelios, relativos al punto más fuerte del audio.
      4. Calcular a qué frecuencia en hercios corresponde cada una de las filas.
    """
    # 1. [n] -> [1025, T] de números COMPLEJOS. Cada columna es una ventana en el
    #    tiempo y cada fila una banda de frecuencia. Las filas son 1 + 2048/2 = 1025
    #    porque la mitad del espectro de una señal real es espejo de la otra y se
    #    descarta. n_fft (2048) es mayor que win_length (1024): cada ventana se
    #    rellena con ceros hasta 2048, lo que no agrega información pero devuelve un
    #    espectro más fino de ver (ver la nota en config.N_FFT).
    stft = librosa.stft(
        np.asarray(muestras, dtype=np.float32),
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
    )

    # 2 y 3. [1025, T] complejo -> [1025, T] real, en decibelios.
    #    np.abs se queda con la magnitud del número complejo (cuánta energía) y tira
    #    la fase (en qué punto de su ciclo iba la onda), que no aporta a la vista.
    #    amplitude_to_db comprime a escala logarítmica, que es como el oído percibe
    #    el volumen: ref=np.max pone el 0 dB en el instante más fuerte de este audio,
    #    así que todos los valores salen negativos y son relativos a ese pico.
    espectrograma_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # 4. [1025]. La fila i corresponde a i * (22050 / 2048) Hz, es decir de 0 Hz a
    #    11 025 Hz repartidos por igual. Hace falta para etiquetar el eje vertical en
    #    hercios en vez de en número de fila.
    frecuencias = librosa.fft_frequencies(sr=config.SR, n_fft=config.N_FFT)

    return espectrograma_db.astype(np.float32), frecuencias.astype(np.float32)


@lru_cache(maxsize=1)
def _transformada_mel():
    """
    Construye una sola vez el `MelSpectrogram` de torchaudio.

    El banco de filtros Mel se calcula al instanciar la transformada, no en cada
    llamada, así que rehacerlo por cada petición sería trabajo repetido y visible
    en los tiempos que reporta la interfaz. De ahí el `lru_cache`: la primera
    llamada lo construye y las demás reciben el mismo objeto ya armado. En la
    práctica son ~1 600 ms la primera vez y ~6 ms las siguientes.

    Los argumentos replican los que XTTS-v2 le pasa a `wav_to_mel_cloning` en el
    camino del Perceiver. Importante: `mel_scale` queda en "htk" (ver la nota en
    config.MEL_ESCALA).

    Qué es un banco de filtros Mel: 80 ventanas triangulares que agrupan las 1025
    bandas del espectro en 80 canales, angostos en los graves y cada vez más anchos
    en los agudos. Imita al oído, que distingue muy bien entre 200 y 300 Hz pero
    apenas entre 8 000 y 8 100 Hz.
    """
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SR,        # 22 050 Hz: define a qué frecuencia real cae cada banda
        n_fft=config.N_FFT,           # 2048: tamaño de la transformada, con relleno de ceros
        win_length=config.WIN_LENGTH, # 1024: la ventana de verdad; fija la resolución real
        hop_length=config.HOP_LENGTH, # 256: cuánto avanza la ventana, o sea cuántos frames salen
        f_min=config.MEL_FMIN,        # 0 Hz: dónde empieza el primer filtro triangular
        f_max=config.MEL_FMAX,        # 8000 Hz: dónde termina el último; arriba de eso se ignora
        n_mels=config.N_MELS,         # 80 filtros, o sea 80 canales de salida
        power=config.MEL_POWER,       # 2 = energía (magnitud al cuadrado), no amplitud
        normalized=False,             # no normalizar por el tamaño de la ventana
        norm=config.MEL_NORM,         # "slaney": cada filtro triangular con área 1
        mel_scale=config.MEL_ESCALA,  # "htk": la fórmula Hz->Mel; librosa usa otra por omisión
    )


def mel_espectrograma(
    muestras: np.ndarray, estadisticas: np.ndarray | None = None
) -> np.ndarray:
    """
    Mel-espectrograma logarítmico de 80 canales, [80, T].

    Este es el tensor que la etapa 2 le entrega al Perceiver Resampler, y la razón
    por la que se calcula con torchaudio en vez de librosa: replica paso a paso lo
    que hace `wav_to_mel_cloning` de XTTS-v2, incluido el logaritmo con piso.

    `estadisticas` son las normas por canal que XTTS-v2 guarda en `mel_stats.pth`
    *dentro del checkpoint*, y por las que divide el mel al final. En la etapa 1
    el checkpoint todavía no se descarga, así que por omisión no se normaliza: la
    visualización no cambia, y la etapa 2 pasará el arreglo [80] cuando lo tenga.

    La receta, paso a paso:
      1. Pasar las muestras de numpy a un tensor de PyTorch.
      2. Aplicar la transformada: ventanear, transformar y agrupar las frecuencias
         en los 80 canales Mel, quedándose con la energía de cada uno.
      3. Comprimir con logaritmo, poniendo antes un piso para no calcular log(0).
      4. Si se recibieron las estadísticas del checkpoint, dividir por ellas.
      5. Devolver el resultado a numpy.
    """
    import torch

    # 1. [n] numpy -> [n] tensor. ascontiguousarray garantiza que los datos estén
    #    seguidos en memoria, que es lo que torch necesita para envolverlos sin
    #    copiarlos; from_numpy comparte la memoria en vez de duplicar el audio.
    señal = torch.from_numpy(np.ascontiguousarray(muestras, dtype=np.float32))

    # no_grad apaga el registro de operaciones para derivadas. Aquí no se entrena
    # nada, así que guardar ese historial sería gastar memoria y tiempo de más.
    with torch.no_grad():
        # 2. [n] -> [80, T] con la ENERGÍA de cada canal Mel. Dentro ocurren tres
        #    cosas encadenadas: ventaneo y FFT (como en espectrograma_stft), elevar
        #    al cuadrado por power=2, y multiplicar por el banco de 80 filtros
        #    triangulares que colapsa las 1025 bandas en 80 canales.
        mel = _transformada_mel()(señal)

        # 3. [80, T] -> [80, T], ahora en escala logarítmica. El clamp sustituye por
        #    1e-5 todo lo que esté por debajo: sin ese piso, un silencio absoluto
        #    valdría log(0) = -infinito y contaminaría el tensor entero. Es también
        #    el motivo de que el mínimo del mel sea siempre log(1e-5) = -11.51.
        mel = torch.log(torch.clamp(mel, min=config.MEL_CLAMP_MIN))

        # 4. División opcional, canal por canal, por las normas del checkpoint.
        if estadisticas is not None:
            # Las normas llegan como arreglo de numpy y hay que convertirlas a
            # tensor, y al mismo tipo del mel, para poder dividir uno entre otro.
            normas = torch.as_tensor(np.asarray(estadisticas), dtype=mel.dtype)
            # unsqueeze(-1) convierte las normas de [80] a [80, 1] para que numpy y
            # torch las repitan a lo largo de los T frames: cada canal se divide
            # entre su propia norma, la misma en todos los instantes.
            mel = mel / normas.unsqueeze(-1)

    # 5. [80, T] tensor -> [80, T] numpy, que es lo que esperan las gráficas y el
    #    resto del proyecto. cpu() es un no-op aquí, pero deja la función lista por
    #    si en la etapa 3 el tensor llega a vivir en la GPU.
    return mel.cpu().numpy()


def contorno_f0(
    muestras: np.ndarray, metodo: str = config.METODO_F0_POR_OMISION
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Contorno de frecuencia fundamental, [T] en Hz.

    La F0 es la frecuencia a la que vibran las cuerdas vocales: la nota de la voz.
    Su recorrido en el tiempo es la entonación, lo que distingue una pregunta de una
    afirmación con las mismas palabras.

    Devuelve `(f0, voz_detectada, confianza)`. En los frames sin voz `f0` vale
    `nan` a propósito: al graficar, eso deja huecos en la línea en lugar de
    hundirla hasta cero, que es lo que hace ver entonaciones que no existen.

    El cálculo usa el mismo `frame_length` y el mismo salto que el mel, así que
    los T frames coinciden con los de las otras representaciones y el contorno se
    puede superponer directamente sobre la forma de onda.

    `metodo` puede ser "pyin" (preciso, detecta voz, lento) o "yin" (rápido, sin
    detección de voz).

    La receta, paso a paso:
      1. Fijar el tipo de las muestras.
      2. Si se pidió "yin": estimar una F0 por frame, sin distinguir si hay voz.
      3. Rechazar cualquier método que no sea uno de los dos conocidos.
      4. Si se pidió "pyin": estimar F0, con detección de voz y confianza.
    """
    # 1. [n] -> [n]. Solo fija el tipo.
    señal = np.asarray(muestras, dtype=np.float32)

    if metodo == "yin":
        # 2. [n] -> [T] en hercios. yin busca, en cada ventana, cada cuántas
        #    muestras la señal se parece más a sí misma: ese periodo que se repite
        #    es el de la onda de la voz, y su inverso es la frecuencia. Acotar la
        #    búsqueda entre 65 y 400 Hz evita confundirse con un armónico.
        f0 = librosa.yin(
            señal,
            fmin=config.F0_MINIMO_HZ,
            fmax=config.F0_MAXIMO_HZ,
            sr=config.SR,
            frame_length=config.N_FFT,
            hop_length=config.HOP_LENGTH,
        )
        # yin no distingue frames sonoros de silencios: entrega un valor siempre.
        # Se rellenan los otros dos arreglos para que la firma de retorno sea la
        # misma con los dos métodos y quien llama no tenga que preguntar cuál se usó.
        voz = np.ones_like(f0, dtype=bool)
        confianza = np.full_like(f0, np.nan, dtype=np.float32)
        return f0.astype(np.float32), voz, confianza

    # 3. Un método mal escrito debe fallar aquí, con un mensaje claro, y no más
    #    adelante con un error incomprensible de librosa.
    if metodo != "pyin":
        raise ValueError(f"Método de F0 no reconocido: {metodo!r}. Usa 'pyin' o 'yin'.")

    # 4. [n] -> tres arreglos de [T]. pyin es la versión probabilística de yin: en vez
    #    de quedarse con el mejor candidato de cada ventana, considera varios con su
    #    probabilidad y decide la secuencia más coherente en conjunto. Por eso puede
    #    además responder si el frame tiene voz o no.
    #      f0        la frecuencia en Hz, o nan donde no detectó voz
    #      voz       True/False por frame
    #      confianza probabilidad de que ese frame sea sonoro
    f0, voz, confianza = librosa.pyin(
        señal,
        fmin=config.F0_MINIMO_HZ,
        fmax=config.F0_MAXIMO_HZ,
        sr=config.SR,
        frame_length=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    return (
        f0.astype(np.float32),
        np.asarray(voz, dtype=bool),
        np.asarray(confianza, dtype=np.float32),
    )


def eje_de_tiempo_frames(n_frames: int) -> np.ndarray:
    """
    Eje de tiempo en segundos para las representaciones por frames.

    El espectrograma, el mel y el F0 no tienen un valor por muestra sino uno por
    ventana, así que su eje horizontal se construye distinto al de la forma de onda:
    el frame i cae en el segundo i * 256 / 22050, porque cada frame avanza 256
    muestras respecto al anterior.
    """
    # [T]. frames_to_time hace esa multiplicación; se usa la función de librosa en
    # vez de escribir la cuenta a mano para que coincida exactamente con el criterio
    # de centrado con el que se calcularon los frames.
    return librosa.frames_to_time(
        np.arange(n_frames), sr=config.SR, hop_length=config.HOP_LENGTH
    ).astype(np.float32)
