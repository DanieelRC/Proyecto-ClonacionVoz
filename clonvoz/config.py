"""Configuración y parámetros de procesamiento de audio para XTTS-v2."""

# Parámetros del mel-espectrograma compatibles con XTTS-v2
SR = 22050  # Frecuencia de muestreo (Hz)
N_FFT = 2048  # Tamaño de FFT (ventana rellena con ceros)
HOP_LENGTH = 256  # Salto entre ventanas (muestras)
WIN_LENGTH = 1024  # Tamaño de la ventana de análisis (muestras)
N_MELS = 80  # Canales del banco de filtros Mel
MEL_FMIN = 0  # Frecuencia mínima Mel (Hz)
MEL_FMAX = 8000  # Frecuencia máxima Mel (Hz)
MEL_POWER = 2.0  # Espectrograma de potencia
MEL_NORM = "slaney"  # Normalización del banco de filtros
MEL_ESCALA = "htk"  # Escala de conversión Hz -> Mel (por defecto en torchaudio)
MEL_CLAMP_MIN = 1e-5  # Umbral mínimo pre-logaritmo: log(clamp(mel, min=1e-5))

# Parámetros de condicionamiento para XTTS-v2
CHUNK_CLONACION_S = 6  # Duración de fragmentos para el Perceiver (s)
MINIMO_CHUNK_S = 0.33  # Duración mínima de fragmento procesable (s)
LONGITUD_MAXIMA_CONDICIONAMIENTO_S = 30  # Duración máxima de referencia (s)
SR_SALIDA = 24000  # Frecuencia de salida del vocoder HiFi-GAN (Hz)
SR_ENCODER_HABLANTE = 16000  # Frecuencia para el codificador de hablante HiFi-GAN (Hz)

# Estimación de frecuencia fundamental (F0)
F0_MINIMO_HZ = 65.0  # Límite inferior de F0 (Hz)
F0_MAXIMO_HZ = 400.0  # Límite superior de F0 (Hz)
METODO_F0_POR_OMISION = "pyin"  # Estimador por defecto ('pyin' o 'yin')

# Validación del audio de referencia
DURACION_MINIMA_S = 3.0  # Duración mínima permitida (s)
DURACION_MAXIMA_S = 30.0  # Duración máxima permitida (s)
DURACION_RECOMENDADA_MINIMA_S = 6.0  # Duración mínima recomendada (s)
DURACION_RECOMENDADA_MAXIMA_S = 15.0  # Duración máxima recomendada (s)
RMS_MINIMO = 1e-3  # Nivel RMS mínimo para descartar silencio
TAMANO_MAXIMO_BYTES = 25 * 1024 * 1024  # Tamaño máximo de carga (25 MB)
FORMATOS_ACEPTADOS = ("WAV", "WAVEX", "RF64")  # Contenedores aceptados por libsndfile


def numero_de_frames(n_muestras: int) -> int:
    """Calcula el número de frames de análisis para una cantidad de muestras."""
    return 1 + n_muestras // HOP_LENGTH
