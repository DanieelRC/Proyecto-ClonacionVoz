"""Etapa 3: tokens de audio del GPT-2 de XTTS-v2, observados paso a paso.

No se reimplementa la generación: se llama a `GPT.generate` oficial y se le pasan
un streamer (recibe cada token), un criterio de parada (cancelación) y un
procesador de logits pasivo (lee la distribución sin modificarla). Los ganchos
en tres capas leen la atención del último token en cada paso.
"""

from __future__ import annotations

import base64
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .identidad import (ModeloNoDisponible, comprobar_archivos, elegir_dispositivo,
                        importar_torch, leer_checkpoint)

IDIOMA = "es"

MUESTREO_XTTS = {"temperature": 0.75, "length_penalty": 1.0, "repetition_penalty": 10.0,
                 "top_k": 50, "top_p": 0.85}
"""Valores por omisión de `Xtts.inference`; se sustituyen por los de config.json si vienen."""


class GeneradorOcupado(RuntimeError):
    """Ya hay una generación en curso: los ganchos son del modelo, no de la petición."""


@dataclass
class Paso:
    """Un token generado y lo que el modelo miraba al elegirlo."""

    paso: int
    token: int
    probabilidad: float
    atencion: dict[int, np.ndarray]
    ms: float


@dataclass
class Generacion:
    """Resultado completo de una corrida de la etapa 3."""

    tokens: np.ndarray
    texto_ids: list[int]
    piezas: list[str]
    motivo: str
    ms_total: float
    prefijo: int
    zonas: dict
    capas: tuple
    tope: int
    pasos: list[Paso] = field(default_factory=list)

    @property
    def ms_por_token(self) -> float:
        return self.ms_total / max(len(self.tokens), 1)

    def como_dict(self) -> dict:
        """Resumen serializable; no incluye las filas de atención."""
        return {
            "tokens": [int(t) for t in self.tokens],
            "texto_ids": self.texto_ids,
            "piezas": self.piezas,
            "motivo": self.motivo,
            "total_tokens": int(len(self.tokens)),
            "ms_total": round(self.ms_total, 1),
            "ms_por_token": round(self.ms_por_token, 2),
            "prefijo": self.prefijo,
            "zonas": self.zonas,
            "capas": [c + 1 for c in self.capas],
            "tope": self.tope,
        }


def fila_cuantizada(fila) -> dict:
    """Fila de atención -> 8 bits en base64 más su máximo, para reconstruir el valor real."""
    maximo = float(fila.max()) if fila.size else 0.0
    escala = fila / maximo if maximo > 0 else np.zeros_like(fila)
    q = np.clip(np.rint(escala * 255), 0, 255).astype(np.uint8)
    return {"max": maximo, "q": base64.b64encode(q.tobytes()).decode("ascii")}


def evento_paso(paso) -> dict:
    """Paso -> diccionario pequeño: nunca el tensor completo (RNF-02)."""
    return {"paso": paso.paso, "token": paso.token,
            "probabilidad": round(paso.probabilidad, 5) if math.isfinite(paso.probabilidad) else None,
            "ms": round(paso.ms, 1),
            "atencion": {str(capa + 1): fila_cuantizada(fila) for capa, fila in paso.atencion.items()}}


class GeneradorTokens:
    """Carga el GPT oficial de XTTS-v2 y genera tokens de audio instrumentados."""

    def __init__(self, carpeta, dispositivo="auto"):
        carpeta = comprobar_archivos(carpeta, ("config.json", "model.pth", "vocab.json"))
        torch = importar_torch()
        from TTS.tts.layers.xtts.gpt import GPT
        from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
        from TTS.tts.models.xtts import XttsArgs

        disp = elegir_dispositivo(torch, dispositivo)
        datos, pesos = leer_checkpoint(carpeta)
        try:
            tokenizador = VoiceBpeTokenizer(vocab_file=str(carpeta / "vocab.json"))
            campos = XttsArgs.__dataclass_fields__
            a = XttsArgs(**{k: v for k, v in datos["model_args"].items() if k in campos})
            # Como Xtts.init_models: estos tres valores salen del vocabulario, no de
            # config.json, que puede traerlos vacíos.
            if tokenizador.tokenizer is not None:
                a.gpt_number_text_tokens = tokenizador.get_number_tokens()
                a.gpt_start_text_token = tokenizador.tokenizer.token_to_id("[START]")
                a.gpt_stop_text_token = tokenizador.tokenizer.token_to_id("[STOP]")
            # Mismos argumentos que Xtts.init_models: nada escrito a mano.
            gpt = GPT(layers=a.gpt_layers, model_dim=a.gpt_n_model_channels,
                      start_text_token=a.gpt_start_text_token, stop_text_token=a.gpt_stop_text_token,
                      heads=a.gpt_n_heads, max_text_tokens=a.gpt_max_text_tokens,
                      max_mel_tokens=a.gpt_max_audio_tokens, max_prompt_tokens=a.gpt_max_prompt_tokens,
                      number_text_tokens=a.gpt_number_text_tokens, num_audio_tokens=a.gpt_num_audio_tokens,
                      start_audio_token=a.gpt_start_audio_token, stop_audio_token=a.gpt_stop_audio_token,
                      use_perceiver_resampler=a.gpt_use_perceiver_resampler,
                      code_stride_len=a.gpt_code_stride_len)
            seleccion = {k[len("gpt."):]: v for k, v in pesos.items() if k.startswith("gpt.")}
            # Igual que Xtts.load_checkpoint: los checkpoints v1 traen las claves de
            # gpt_inference y solo cargan después de crearlo.
            try:
                gpt.load_state_dict(seleccion, strict=True)
            except RuntimeError:
                gpt.init_gpt_for_inference(kv_cache=True)
                gpt.load_state_dict(seleccion, strict=True)
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            raise ModeloNoDisponible(
                "No se pudo cargar el GPT-2 de XTTS-v2. Comprueba config.json, model.pth "
                "y vocab.json. Detalle: " + str(exc)
            ) from exc
        muestreo = {clave: datos.get(clave, omision) for clave, omision in MUESTREO_XTTS.items()}
        self._preparar(torch, gpt, tokenizador, muestreo, disp)

    @classmethod
    def desde_modulos(cls, gpt, tokenizador, muestreo=None, dispositivo="cpu"):
        """Construye el generador con módulos ya creados (pruebas y scripts)."""
        objeto = cls.__new__(cls)
        objeto._preparar(importar_torch(), gpt, tokenizador,
                         {**MUESTREO_XTTS, **(muestreo or {})}, dispositivo)
        return objeto

    def _preparar(self, torch, gpt, tokenizador, muestreo, dispositivo):
        gpt.init_gpt_for_inference(kv_cache=True)
        # SDPA no devuelve los pesos de atención; eager sí, con los mismos tokens.
        gpt.gpt.set_attn_implementation("eager")
        gpt.gpt._attn_implementation = "eager"
        gpt.to(dispositivo).eval().requires_grad_(False)
        self.torch = torch
        self.gpt = gpt
        self.tokenizador = tokenizador
        self.muestreo = muestreo
        self.dispositivo = dispositivo
        n = len(gpt.gpt.h)
        self.capas = tuple(sorted({0, (n - 1) // 2, n - 1}))
        self.tope = gpt.max_gen_mel_tokens
        self.max_tokens_texto = gpt.max_text_tokens - 2
        limites = getattr(tokenizador, "char_limits", {})
        self.limite_caracteres = limites.get(IDIOMA, 239)
        self.cerrojo = threading.Lock()

    def tokenizar(self, texto) -> tuple[list[int], list[str]]:
        """Texto -> ids BPE, con el mismo preprocesado que `Xtts.inference`."""
        limpio = (texto or "").strip()
        if not limpio:
            raise ValueError("Escribe el texto que quieres generar.")
        if len(limpio) > self.limite_caracteres:
            raise ValueError(f"El texto tiene {len(limpio)} caracteres; el máximo en español "
                             f"es {self.limite_caracteres}.")
        ids = [int(i) for i in self.tokenizador.encode(limpio.lower(), lang=IDIOMA)]
        if len(ids) >= self.max_tokens_texto:
            raise ValueError("El texto produce demasiados tokens para el modelo. Acórtalo.")
        return ids, [self._pieza(i) for i in ids]

    def evento_inicio(self, id_corrida, texto, ids, piezas, determinista, semilla) -> dict:
        """Lo que la vista necesita antes del primer token: fichas, zonas y capas."""
        prefijo = 32 + len(ids) + 3
        return {"id_corrida": id_corrida, "texto": texto.strip(), "ids": ids, "piezas": piezas,
                "prefijo": prefijo,
                "zonas": {"voz": [0, 32], "texto": [32, 32 + len(ids) + 2], "audio": [prefijo - 1, None]},
                "capas": [c + 1 for c in self.capas], "total_capas": len(self.gpt.gpt.h),
                "tope": self.tope, "determinista": determinista, "semilla": semilla,
                "muestreo": None if determinista else self.muestreo}

    def _pieza(self, identificador) -> str:
        interno = getattr(self.tokenizador, "tokenizer", None)
        pieza = interno.id_to_token(identificador) if interno is not None else None
        if pieza is None:
            pieza = self.tokenizador.decode([identificador])
        return pieza.replace("[SPACE]", "␣")

    def generar(self, vectores, texto, *, al_paso=None, detener=None,
                determinista=False, semilla=0) -> Generacion:
        """Genera tokens de audio a partir de la identidad [32, 1024] y un texto.

        1. Validar la identidad y tokenizar el texto.
        2. Colgar ganchos en tres capas y preparar streamer, observador y parada.
        3. Llamar a `GPT.generate` oficial; cada token se entrega a `al_paso`.
        4. Retirar los ganchos pase lo que pase.
        """
        torch = self.torch
        vectores = np.asarray(vectores, dtype=np.float32)
        if vectores.shape != (32, self.gpt.model_dim) or not np.isfinite(vectores).all():
            raise ValueError(f"Se esperaba una identidad finita de forma [32, {self.gpt.model_dim}].")
        ids, piezas = self.tokenizar(texto)

        # [32 voz | inicio_texto, texto, fin_texto | inicio_audio], luego un token por paso.
        prefijo = 32 + len(ids) + 2 + 1
        zonas = {"voz": [0, 32], "texto": [32, 32 + len(ids) + 2], "audio": [prefijo - 1, None]}

        if not self.cerrojo.acquire(blocking=False):
            raise GeneradorOcupado("Ya hay una generación en curso. Espera a que termine.")
        try:
            return self._generar(torch, vectores, ids, piezas, prefijo, zonas,
                                 al_paso, detener, determinista, semilla)
        finally:
            self.cerrojo.release()

    def _generar(self, torch, vectores, ids, piezas, prefijo, zonas,
                 al_paso, detener, determinista, semilla):
        from transformers import (LogitsProcessor, LogitsProcessorList, StoppingCriteria,
                                  StoppingCriteriaList)
        from transformers.generation.streamers import BaseStreamer

        ultima_fila, distribucion, pasos, tokens = {}, {}, [], []
        reloj = {"inicio": time.perf_counter()}
        reloj["ultimo"] = reloj["inicio"]

        def gancho(capa):
            def capturar(_modulo, _entrada, salida):
                # [1, cabezas, consultas, claves] -> fila del último token, promedio de cabezas.
                if salida[1] is not None:
                    ultima_fila[capa] = salida[1][0, :, -1, :].float().mean(dim=0).cpu().numpy()
            return capturar

        class Observador(LogitsProcessor):
            def __call__(self, _ids, puntajes):
                distribucion["p"] = torch.softmax(puntajes[0].float(), dim=-1)
                return puntajes

        class Parada(StoppingCriteria):
            def __call__(self, entrada, _puntajes, **_kwargs):
                activa = detener is not None and detener.is_set()
                return torch.full((entrada.shape[0],), activa, dtype=torch.bool, device=entrada.device)

        class Flujo(BaseStreamer):
            def __init__(self):
                self.prompt = True

            def put(self, valor):
                if self.prompt:  # generate entrega primero el prompt completo
                    self.prompt = False
                    return
                token = int(valor.reshape(-1)[0])
                ahora = time.perf_counter()
                probabilidad = distribucion.get("p")
                paso = Paso(paso=len(tokens), token=token,
                            probabilidad=float(probabilidad[token]) if probabilidad is not None else float("nan"),
                            atencion=dict(ultima_fila), ms=(ahora - reloj["ultimo"]) * 1000)
                reloj["ultimo"] = ahora
                tokens.append(token)
                pasos.append(paso)
                if al_paso is not None:
                    al_paso(paso)

            def end(self):
                pass

        argumentos = {"repetition_penalty": self.muestreo["repetition_penalty"],
                      "length_penalty": self.muestreo["length_penalty"], "num_beams": 1}
        if determinista:
            argumentos["do_sample"] = False
        else:
            argumentos.update(do_sample=True, temperature=self.muestreo["temperature"],
                              top_k=self.muestreo["top_k"], top_p=self.muestreo["top_p"])

        ganchos = [self.gpt.gpt.h[c].attn.register_forward_hook(gancho(c)) for c in self.capas]
        try:
            torch.manual_seed(int(semilla))
            cond = torch.from_numpy(vectores)[None].to(self.dispositivo)
            texto = torch.IntTensor(ids)[None].to(self.dispositivo)
            with torch.inference_mode():
                codigos = self.gpt.generate(
                    cond, texto, streamer=Flujo(), output_attentions=False,
                    logits_processor=LogitsProcessorList([Observador()]),
                    stopping_criteria=StoppingCriteriaList([Parada()]), **argumentos)
        finally:
            for g in ganchos:
                g.remove()

        salida = codigos[0].cpu().numpy().astype(np.int64)
        if detener is not None and detener.is_set():
            motivo = "cancelada"
        elif len(salida) and salida[-1] == self.gpt.stop_audio_token:
            motivo = "parada"
        else:
            motivo = "tope"
        return Generacion(tokens=salida, texto_ids=ids, piezas=piezas, motivo=motivo,
                          ms_total=(time.perf_counter() - reloj["inicio"]) * 1000,
                          prefijo=prefijo, zonas=zonas, capas=self.capas, tope=self.tope,
                          pasos=pasos)
