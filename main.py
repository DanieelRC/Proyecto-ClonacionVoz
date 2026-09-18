"""
Primer avance de código (versión mínima)
Clona una voz a partir de un audio de referencia y genera un audio nuevo a partir de texto con ella,
usando un modelo preentrenado de código abierto (XTTS-v2, zero-shot).
"""

from TTS.api import TTS

modelo = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

modelo.tts_to_file(
    text="Hola, esta es una prueba de clonación de voz.",
    speaker_wav="audios/muestra_hablante.wav",
    language="es",
    file_path="audios/resultado.wav",
)