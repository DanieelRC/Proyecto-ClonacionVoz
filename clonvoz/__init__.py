"""
Aplicación de clonación de voz con visualización en vivo de la arquitectura XTTS-v2.

Paquete principal del proyecto. Cada etapa del plan de desarrollo agrega sus
propios módulos aquí sin mover los anteriores:

    Etapa 1  config, audio, caracteristicas, graficas, analisis, cli, web
    Etapa 2  (pendiente) codificador de hablante: Perceiver Resampler
    Etapa 3  (pendiente) backbone generativo: GPT-2 autoregresivo
    Etapa 4  (pendiente) decodificación DVAE + vocoder HiFi-GAN
    Etapa 5  (pendiente) integración en tiempo real: hooks y WebSocket
    Etapa 6  (pendiente) frontend de exposición y modo grabado
"""

ETAPA = 1
