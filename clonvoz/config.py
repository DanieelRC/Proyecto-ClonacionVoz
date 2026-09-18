"""
Parámetros de procesamiento de audio, en un solo lugar.

Los parámetros del mel-espectrograma replican los que usa XTTS-v2 para alimentar
al Perceiver Resampler. Esto no es un detalle estético: en la etapa 2 el
mel-espectrograma [80, T] que calcula la etapa 1 se le entrega tal cual al
Perceiver, así que cualquier diferencia en estos números rompe esa etapa, no
esta. El síntoma aparecería allá disfrazado de "el modelo no funciona".

DE DÓNDE SALEN ESTOS VALORES
Se leyeron del código instalado de coqui-tts 0.27.5, no de memoria. La función es
`wav_to_mel_cloning` (`TTS/tts/models/xtts.py`), y lo que manda son los
argumentos con los que la llama `XTTS.get_gpt_cond_latents`, no los valores por
omisión de su firma, que son otros. Ojo con dos confusiones fáciles:

  * La clase `TorchMelSpectrogram` (`TTS/tts/layers/tortoise/arch_utils.py`) tiene
    n_fft=1024 y parece la candidata obvia, pero XTTS-v2 **no la usa** para este
    camino: la usan Tortoise y el entrenador del GPT.
  * `wav_to_mel_cloning` tiene n_fft=4096 por omisión en su firma, y el camino sin
    Perceiver (XTTS-v1) sí usa 4096/1024/4096. El camino del Perceiver, que es el
    de XTTS-v2, pasa 2048/256/1024.

Ningún otro módulo debe redefinir estas constantes: todos las importan de aquí.
Para comprobar que siguen coincidiendo con el paquete instalado:

    python scripts/verificar_compatibilidad_mel.py audios/muestra_hablante.wav
"""

# --------------------------------------------------------------------------
# Parámetros del mel-espectrograma (deben coincidir con XTTS-v2)
# --------------------------------------------------------------------------

SR = 22050
"""Frecuencia de muestreo del camino de condicionamiento de XTTS-v2, en Hz."""

N_FFT = 2048
"""
Tamaño de la FFT.

No es igual a WIN_LENGTH a propósito: XTTS-v2 analiza ventanas de 1024 muestras y
las rellena con ceros hasta 2048 antes de transformarlas. Eso no agrega
información, solo interpola el eje de frecuencia: la resolución real la fija la
ventana, y la FFT más larga produce un espectro más fino de ver.
"""

HOP_LENGTH = 256
"""Salto entre ventanas consecutivas, en muestras (11.6 ms a 22 050 Hz)."""

WIN_LENGTH = 1024
"""Tamaño de la ventana de análisis, en muestras (46.4 ms a 22 050 Hz)."""

N_MELS = 80
"""Canales del banco de filtros Mel (`n_mel_channels`): los 80 que pide el proyecto."""

MEL_FMIN = 0
"""Frecuencia mínima del banco de filtros Mel, en Hz."""

MEL_FMAX = 8000
"""Frecuencia máxima del banco de filtros Mel, en Hz."""

MEL_POWER = 2.0
"""Espectrograma de potencia (magnitud al cuadrado), no de amplitud."""

MEL_NORM = "slaney"
"""Normalización del banco de filtros: área unitaria por filtro."""

MEL_ESCALA = "htk"
"""
Escala de conversión Hz -> Mel.

Aquí está la trampa de esta etapa. XTTS-v2 construye su `MelSpectrogram` de
torchaudio pasando `norm="slaney"` pero dejando `mel_scale` en su valor por
omisión, que en torchaudio es "htk". librosa, en cambio, usa por omisión la
escala slaney (`htk=False`). Son bancos de filtros distintos y producen mels
distintos. Por eso el mel se calcula con torchaudio y no con librosa: replicar
la fórmula exacta cuesta menos que depurarla en la etapa 2.
"""

MEL_CLAMP_MIN = 1e-5
"""Piso antes del logaritmo: XTTS-v2 aplica log(clamp(mel, min=1e-5))."""

# --------------------------------------------------------------------------
# Cómo consume XTTS-v2 este mel (información para la etapa 2)
# --------------------------------------------------------------------------

CHUNK_CLONACION_S = 6
"""
XTTS-v2 no pasa el mel completo por el Perceiver de una sola vez.

Parte el audio de referencia en trozos de 6 segundos, saca un embedding de estilo
de cada uno y promedia los resultados. La etapa 2 tendrá que reproducir ese
troceado; la etapa 1 calcula el mel completo, que es lo que hay que visualizar.
"""

MINIMO_CHUNK_S = 0.33
"""Trozos más cortos que esto los descarta XTTS-v2 en vez de procesarlos."""

LONGITUD_MAXIMA_CONDICIONAMIENTO_S = 30
"""
XTTS-v2 recorta el audio de referencia a 30 s antes de calcular el mel.

Por eso DURACION_MAXIMA_S vale lo mismo: aceptar más sería aceptar audio que el
modelo va a tirar de todos modos.
"""

SR_SALIDA = 24000
"""
Frecuencia de muestreo del audio que genera el vocoder HiFi-GAN, en Hz.

No se usa en la etapa 1; queda documentada aquí porque es la de la etapa 4 y es
distinta de SR: XTTS-v2 analiza la referencia a 22 050 Hz y sintetiza a 24 000 Hz.
"""

SR_ENCODER_HABLANTE = 16000
"""
Frecuencia a la que XTTS-v2 remuestrea la referencia para el codificador de
hablante del HiFi-GAN, en Hz.

Tampoco se usa en la etapa 1. Se anota porque es fácil confundirse: la referencia
recorre dos caminos con frecuencias distintas —22 050 Hz hacia el Perceiver y
16 000 Hz hacia este codificador— y el mel de esta etapa sirve solo para el primero.
"""

# --------------------------------------------------------------------------
# Estimación del contorno de frecuencia fundamental (F0)
# --------------------------------------------------------------------------

F0_MINIMO_HZ = 65.0
"""Piso de búsqueda de F0, en Hz. Cubre voces masculinas graves (~C2)."""

F0_MAXIMO_HZ = 400.0
"""Techo de búsqueda de F0, en Hz. Cubre voz hablada adulta con margen."""

METODO_F0_POR_OMISION = "pyin"
"""
Estimador de F0 por omisión.

"pyin" es probabilístico: más preciso y además reporta qué frames tienen voz,
pero es el paso más lento de toda la etapa. "yin" es notablemente más rápido y
no distingue frames sonoros de silencios, así que dibuja un contorno continuo
incluso donde no hay voz. El tiempo real de cada uno se reporta en la interfaz
(RF-20), para decidir con datos y no con suposiciones.
"""

# --------------------------------------------------------------------------
# Validación del audio de referencia (RF-03)
# --------------------------------------------------------------------------

DURACION_MINIMA_S = 3.0
"""Por debajo de esto el audio se rechaza: no alcanza para caracterizar una voz."""

DURACION_MAXIMA_S = 30.0
"""Por encima de esto el audio se rechaza: alarga el análisis sin aportar nada."""

DURACION_RECOMENDADA_MINIMA_S = 6.0
"""Mínimo recomendado. Entre DURACION_MINIMA_S y este valor se acepta con advertencia."""

DURACION_RECOMENDADA_MAXIMA_S = 15.0
"""Máximo recomendado: XTTS-v2 clona mejor con referencias de 6 a 15 segundos."""

RMS_MINIMO = 1e-3
"""Nivel RMS por debajo del cual el audio se considera silencio o casi silencio."""

TAMANO_MAXIMO_BYTES = 25 * 1024 * 1024
"""Tamaño máximo aceptado en la carga (25 MB), holgado para 30 s de WAV."""

FORMATOS_ACEPTADOS = ("WAV", "WAVEX", "RF64")
"""
Contenedores aceptados, según los reporta libsndfile.

Se valida el contenedor real y no la extensión del nombre: libsndfile también
sabe leer MP3, así que un archivo renombrado a .wav se abriría sin error y hay
que rechazarlo explícitamente para cumplir RF-02.
"""


def numero_de_frames(n_muestras: int) -> int:
    """
    Frames que produce el análisis para una señal de `n_muestras`.

    Con centrado de ventanas (el valor por omisión tanto en torchaudio como en
    librosa) las tres representaciones por frames —espectrograma, mel y F0—
    comparten esta misma longitud T, de modo que comparten eje temporal y el
    contorno de F0 se puede superponer a las demás sin reescalar nada.
    """
    return 1 + n_muestras // HOP_LENGTH
