"""Script de síntesis mínima con XTTS-v2 (TTS zero-shot a partir de audio de referencia)."""

from TTS.api import TTS

modelo = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

modelo.tts_to_file(
    text="Hola, esta es una prueba de clonación de voz.",
    speaker_wav="audios/muestra_hablante.wav",
    language="es",
    file_path="audios/resultado.wav",
)