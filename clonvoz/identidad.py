"""Etapa 2: módulos oficiales de XTTS-v2, sin construir GPT-2 ni HiFi-GAN.

La captura observa el dropout de atención en modo eval: recibe exactamente las
probabilidades usadas por el modelo. No sustituye ninguna operación neuronal.

Recorrido de esta etapa (T es el número de frames del mel completo):

    Resultado de etapa 1: log-mel completo [80,T]
        -> normalización con mel_stats [80]
        -> ConditioningEncoder [1,1024,T]
        -> cambiar orden de ejes [1,T,1024]
        -> Perceiver [1,32,1024]
        -> quitar dimensión de lote -> identidad/estilo [32,1024]

Un tensor es un arreglo de números con varias dimensiones. El 1 de las formas
anteriores es el tamaño del lote: procesamos una referencia por vez. T cambia
con la duración, mientras que las 32 consultas del Perceiver son fijas. Esa es
la compresión que interesa enseñar: muchos instantes pasan a 32 vectores.

Hay dos responsabilidades separadas: CodificadorIdentidad conserva los pesos y
procesa un mel; identidad_desde_mel empaqueta los resultados y sus tiempos.
Ninguna necesita Flask ni escribe audios o resultados en disco.
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np



class ModeloNoDisponible(RuntimeError):
    """Problema de instalación o pesos que se puede explicar al usuario."""


@dataclass
class Identidad:
    """Paquete de resultados de una referencia; no contiene sus muestras WAV.

    dataclass crea el constructor para reunir estos campos sin escribir a mano
    una asignación por campo. No ejecuta el modelo ni transforma los tensores.
    """

    id_corrida: str
    # Salida sobre el mel completo. Cada fila es una consulta aprendida.
    vectores: np.ndarray
    # [32,T]. Atención al contexto codificado a partir del mel completo.
    atencion: np.ndarray
    # [1,32,32]. Parte de la atención dirigida a las propias consultas.
    atencion_consultas: np.ndarray
    # Intervalos de audio y columnas de la imagen para ubicar cada fragmento.
    fragmentos: list[dict]
    tiempos_ms: dict
    advertencias: list[str]
    dispositivo: str
    # Diagnóstico opcional de ambas capas; los vectores originales no cambian.
    capas_atencion: list | None = None

    def como_dict(self):
        """Construye el resumen pequeño que se imprime o se envía como JSON.

        shape cuenta filas y columnas; min/max/mean/std resumen los valores.
        float convierte números NumPy a números que el serializador JSON admite.
        Las gráficas se añaden en el servidor, no en este módulo de inferencia.
        """
        # Los tensores solo se exportan por petición expresa en la CLI.
        # Cada fila reparte su atención entre consultas (primeras 32 columnas)
        # y audio (las restantes). Sumamos cada grupo y promediamos las filas.
        # Las cabezas ya se promediaron al capturar; multiplicar por 100 solo
        # cambia la unidad de presentación. No normalizamos ni alteramos pesos.
        capas = self.capas_atencion or [np.concatenate(
            (self.atencion_consultas[0], self.atencion), axis=1)]
        resumen_atencion = []
        for indice, capa in enumerate(capas):
            resumen_atencion.append({
                "capa": ("Primera capa" if indice == 0 else "Última capa")
                        if self.capas_atencion else "Última capa",
                "audio_porcentaje": float(capa[:, 32:].sum(axis=1, dtype=np.float64).mean() * 100),
                "consultas_porcentaje": float(capa[:, :32].sum(axis=1, dtype=np.float64).mean() * 100),
            })
        return {
            "resumen_atencion": resumen_atencion,
            "id_corrida": self.id_corrida,
            "formas": {"identidad": list(self.vectores.shape),
                       "atencion": list(self.atencion.shape)},
            "fragmentos": self.fragmentos,
            "tiempos_ms": {k: round(v, 2) for k, v in self.tiempos_ms.items()},
            "advertencias": self.advertencias,
            "dispositivo": self.dispositivo,
            "captura": "Última capa del Perceiver, promedio de sus 8 cabezas; sin renormalizar.",
            "estadisticas": {"minimo": float(self.vectores.min()),
                             "maximo": float(self.vectores.max()),
                             "media": float(self.vectores.mean()),
                             "desviacion": float(self.vectores.std())},
        }


def carpeta_modelo():
    """Busca CLONVOZ_MODELO; si no está definida, usa modelos/xtts_v2.

    Es una ruta relativa a la carpeta desde donde se ejecuta el programa. Por
    eso los ejemplos del README se ejecutan desde la raíz del proyecto.
    """
    return Path(os.environ.get("CLONVOZ_MODELO", "modelos/xtts_v2"))


class CodificadorIdentidad:
    """Carga una vez los módulos preentrenados y los reutiliza entre referencias.

    No entrenamos una red nueva: importamos las clases oficiales y colocamos en
    ellas sus pesos aprendidos. Los pesos son números que determinan cómo
    transformar la entrada; sin ellos, la estructura sola no caracteriza voces.
    """

    def __init__(self, carpeta, dispositivo="auto"):
        """Preparación paso a paso:

        1. Comprobar que existen la configuración y el checkpoint.
        2. Importar las clases oficiales y elegir CPU o GPU.
        3. Verificar las dimensiones del modelo configurado.
        4. Leer los pesos y separar los de nuestros dos módulos.
        5. Cargarlos estrictamente y activar el modo de evaluación.
        6. Conservar las estadísticas Mel y preparar el bloqueo de concurrencia.
        """
        # 1. config.json describe el modelo; model.pth contiene los pesos.
        carpeta = Path(carpeta)
        if not all((carpeta / nombre).is_file() for nombre in ("config.json", "model.pth")):
            raise ModeloNoDisponible(
                "Faltan config.json y/o model.pth de XTTS-v2. "
                "Consulta la preparación de la etapa 2 en el README."
            )
        # 2. Los imports pesados ocurren aquí, solo al pedir la etapa 2.
        # Importar una clase de XTTS no construye una instancia del modelo completo.
        try:
            import torch
            from TTS.tts.layers.tortoise.autoregressive import ConditioningEncoder
            from TTS.tts.layers.xtts.perceiver_encoder import PerceiverResampler
            from TTS.config.shared_configs import BaseDatasetConfig
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
        except (ImportError, OSError) as exc:
            raise ModeloNoDisponible(
                "No se pudieron cargar PyTorch y los módulos de coqui-tts. "
                "Usa el entorno de la etapa 2 indicado en el README."
            ) from exc

        self.torch = torch
        # "auto" usa CUDA si PyTorch puede verla; forzar "cuda" sin soporte es un
        # error explícito. El entorno CPU del README puede ejecutar esta etapa.
        if dispositivo not in ("auto", "cpu", "cuda"):
            raise ModeloNoDisponible("El dispositivo debe ser auto, cpu o cuda.")
        self.dispositivo = ("cuda" if torch.cuda.is_available() else "cpu") if dispositivo == "auto" else dispositivo
        if self.dispositivo == "cuda" and not torch.cuda.is_available():
            raise ModeloNoDisponible("CUDA no está disponible. Selecciona cpu o auto.")

        try:
            # 3. El encoder usa 16 cabezas, pero el Perceiver usa 8: son módulos
            # distintos. La configuración debe corresponder al modelo esperado.
            datos = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
            args = datos["model_args"]
            if (args["gpt_n_model_channels"] != 1024 or args["gpt_n_heads"] != 16
                    or not args["gpt_use_perceiver_resampler"]):
                raise ValueError("La configuración no corresponde al Perceiver de XTTS-v2.")
            # 4. Leer el checkpoint completo es distinto de construir el modelo completo.
            # mmap evita copiar todos sus pesos a RAM; solo conservamos estos módulos.
            # El checkpoint oficial incluye cuatro clases de configuración además
            # de tensores. Se permiten esas clases conocidas, no pickle arbitrario.
            with torch.serialization.safe_globals([XttsConfig, XttsArgs, XttsAudioConfig, BaseDatasetConfig]):
                checkpoint = torch.load(carpeta / "model.pth", map_location="cpu", weights_only=True, mmap=True)
            pesos = checkpoint["model"]
            # Algunos checkpoints de entrenamiento anteponen "xtts." a las claves.
            # Se quita solo ese prefijo para poder buscar los nombres oficiales.
            pesos = {k.removeprefix("xtts."): v for k, v in pesos.items()}
            self.encoder = ConditioningEncoder(80, 1024, num_attn_heads=16)
            # depth=2: dos bloques de atención. num_latents=32: 32 consultas
            # aprendidas. dim=1024: longitud del vector de cada consulta.
            # Desactivar flash hace observable la atención explícita del módulo.
            self.perceiver = PerceiverResampler(dim=1024, depth=2, dim_context=1024,
                                               num_latents=32, dim_head=64, heads=8,
                                               ff_mult=4, use_flash_attn=False)
            for modulo, prefijo in ((self.encoder, "gpt.conditioning_encoder."),
                                    (self.perceiver, "gpt.conditioning_perceiver.")):
                # 5. Ejemplo: gpt.conditioning_encoder.init.weight -> init.weight.
                # El módulo local espera nombres relativos, sin la ruta del padre.
                seleccion = {k[len(prefijo):]: v for k, v in pesos.items() if k.startswith(prefijo)}
                # strict=True impide continuar con pesos incompletos o aleatorios.
                modulo.load_state_dict(seleccion, strict=True)
                # to mueve los pesos al dispositivo; eval desactiva el dropout de
                # entrenamiento; requires_grad_(False) evita calcular gradientes.
                modulo.to(self.dispositivo).eval().requires_grad_(False)
            # 6. [80]: un divisor por canal Mel. copy independiza este arreglo
            # del almacenamiento del checkpoint, que no necesitamos conservar.
            self.normas = pesos["mel_stats"].float().cpu().numpy().copy()
            # Son estadísticas del log-mel: los valores negativos son válidos.
            if self.normas.shape != (80,) or not np.isfinite(self.normas).all() or (self.normas == 0).any():
                raise ValueError("Las estadísticas Mel del checkpoint no son válidas.")
        except (KeyError, ValueError, RuntimeError, OSError, EOFError, pickle.UnpicklingError) as exc:
            raise ModeloNoDisponible(
                "No se pudo cargar el checkpoint compatible de XTTS-v2. "
                "Comprueba los archivos oficiales y sus versiones. Detalle: " + str(exc)
            ) from exc
        self.cerrojo = threading.Lock()

    def procesar_mel(self, mel, *, diagnostico=None):
        """Recibe el log-mel SIN normalizar de etapa 1: [80,T].

        Devuelve vectores [32,1024], atención al audio [32,T] y atención
        a las consultas [32,32]. Estas dos últimas juntas suman uno por fila.

        Pasos:
          1. Validar los 80 canales, al menos un frame y números finitos.
          2. Preparar una función que observe la atención mediante un hook.
          3. Normalizar el mel y pasarlo por encoder y Perceiver.
          4. Retirar el hook incluso si la inferencia falla.
          5. Comprobar las formas y separar atención a consultas y a audio.
        """
        # 1. float32 es el formato numérico de los pesos y reduce uso de memoria
        # frente a float64. isfinite descarta NaN e infinitos antes de la red.
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] == 0 or not np.isfinite(mel).all():
            raise ValueError("Se esperaba un mel finito de forma [80, T], con T mayor que cero.")
        torch = self.torch
        capturas = []

        def capturar(_modulo, _entrada, salida):
            """Observa una activación real; no devuelve una salida de reemplazo."""
            # 2. Un hook es una función llamada cuando termina un módulo.
            # [1,8,32,32+T] -> [32,32+T]. Las primeras 32 claves son consultas.
            # detach separa del registro de gradientes; mean promedia cabezas;
            # [0] quita el lote; cpu permite convertir a NumPy incluso usando GPU.
            capturas.append(salida.detach().mean(dim=1)[0].cpu().numpy().copy())

        # El servidor puede atender dos solicitudes a la vez. El hook de una
        # referencia nunca debe capturar las activaciones de la otra.
        with self.cerrojo, torch.inference_mode():
            # layers[-1] es el último bloque; [0] es su atención (el [1] es la
            # transformación posterior). En eval el dropout deja pasar los pesos
            # sin cambiarlos, así observamos los que se usan para combinar valores.
            # Observamos ambas capas en orden, sin sustituir sus operaciones.
            hooks = [capa[0].attend.attn_dropout.register_forward_hook(capturar)
                     for capa in self.perceiver.layers]
            try:
                # 3. [80] -> [80,1] para dividir todas las columnas de cada canal
                # por su propia estadística. unsqueeze(0) añade el lote: [1,80,T].
                entrada = torch.from_numpy(mel / self.normas[:, None]).unsqueeze(0).to(self.dispositivo)
                contexto = self.encoder(entrada).transpose(1, 2)
                # El encoder entrega [1,1024,T]; transpose lo vuelve [1,T,1024].
                # El Perceiver devuelve [1,32,1024], sin importar la longitud T.
                vectores = self.perceiver(contexto)[0].cpu().numpy().copy()
            finally:
                # 4. Evita dejar capturadores acumulados entre referencias.
                for hook in hooks:
                    hook.remove()
        # 5. Fallar con un mensaje si cambia la arquitectura del paquete instalado.
        if len(capturas) != 2 or any(c.shape != (32, 32 + mel.shape[1]) for c in capturas):
            raise ModeloNoDisponible("La atención del módulo instalado tiene una forma inesperada.")
        if vectores.shape != (32, 1024) or not np.isfinite(vectores).all():
            raise ModeloNoDisponible("El modelo produjo vectores inválidos.")
        # [:,32:] conserva todas las filas y las columnas de audio. No volvemos a
        # dividir por la suma: eso ocultaría cuánto peso se dirigió a las consultas.
        if diagnostico is not None:
            diagnostico.extend(capturas)
        # Conservamos la salida pública de la última capa para CLI y verificaciones.
        return vectores, capturas[-1][:, 32:], capturas[-1][:, :32]


def identidad_desde_mel(mel, modelo, id_corrida, duracion_s, advertencias=()):
    """Etapa 2 recibe la salida de etapa 1: no acepta WAV ni calcula otro mel.

    1. Pasar la misma matriz [80,T] a procesar_mel. Allí se normaliza con los
       pesos reales y se ejecutan el encoder y el Perceiver oficiales.
    2. Conservar el identificador de etapa 1 para relacionar ambos resultados.
    3. Empaquetar atención [32,T] y vectores [32,1024], sin promediar fragmentos.

    Este es el recorrido completo del mel que pide el prototipo de etapa 2.
    Es distinto del promedio por trozos de get_gpt_cond_latents en síntesis XTTS:
    aquí comprobamos equivalencia con get_style_emb sobre el mismo mel completo.
    """
    inicio = time.perf_counter()
    capas = []
    if isinstance(modelo, CodificadorIdentidad):
        vectores, atencion, propias = modelo.procesar_mel(mel, diagnostico=capas)
    else:
        vectores, atencion, propias = modelo.procesar_mel(mel)
    transcurrido = (time.perf_counter() - inicio) * 1000
    # La gráfica reutiliza el formato de intervalos, ahora con un solo intervalo:
    # el audio completo. No hay cortes, recálculo del mel ni un promedio oculto.
    intervalos = [{"inicio_s": 0.0, "fin_s": duracion_s,
                   "columna_inicio": 0, "frames": int(mel.shape[1]),
                   "masa_atencion_audio": float(atencion.sum(axis=1).mean())}]
    return Identidad(id_corrida, vectores, atencion, propias[None, :, :], intervalos,
                     {"codificacion": transcurrido, "total": transcurrido},
                     list(advertencias), modelo.dispositivo, capas or None)


def analizar_identidad(fuente, modelo):
    """Conveniencia para terminal: encadena etapa 1 -> etapa 2 una sola vez.

    La web usa identidad_desde_mel sobre su resultado guardado. Aquí no existe
    sesión web: ejecutamos analizar y entregamos su mel directamente al modelo.
    """
    from .analisis import analizar

    etapa1 = analizar(fuente)
    return identidad_desde_mel(etapa1.mel, modelo, etapa1.id_corrida,
                               etapa1.audio.duracion_s, etapa1.validacion.advertencias)
