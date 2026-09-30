"""Rutas de la etapa 3: generación en vivo (SSE) y audio crudo por fases."""

import json
import os
import queue
import threading

import numpy as np
from flask import Blueprint, Response, current_app, jsonify, request, session

from .. import caracteristicas
from ..audio import AudioInvalido, cargar_audio
from ..fases import (FASES, ReconstructorFases, a_wav, audio_desde_mel_referencia)
from ..generacion import GeneradorOcupado, GeneradorTokens, evento_paso
from ..identidad import ModeloNoDisponible, carpeta_modelo
from .identidad import fabrica_identidad


def fabrica_generador():
    """Construye el GPT-2 de la etapa 3 con la carpeta y el dispositivo configurados."""
    return GeneradorTokens(carpeta_modelo(), os.environ.get("CLONVOZ_DISPOSITIVO", "auto"))


def formato_sse(tipo, datos):
    return f"event: {tipo}\ndata: {json.dumps(datos, ensure_ascii=False)}\n\n"


def crear_rutas_generacion(memoria):
    """Crea el blueprint de la etapa 3 sobre la memoria compartida con las etapas 1 y 2."""
    rutas = Blueprint("generacion", __name__)

    def entrada_de_sesion(identificador):
        if not isinstance(identificador, str) or not identificador:
            raise KeyError(identificador)
        return memoria.obtener(session.get("propietario"), identificador)

    @rutas.post("/api/generar")
    def generar():
        """Genera tokens de audio con la identidad de la etapa 2 y los transmite en vivo.

        1. Validar la petición y recuperar la identidad guardada por la etapa 2.
        2. Cargar el GPT-2 (una vez) y tokenizar para rechazar textos inválidos antes de transmitir.
        3. Generar en un hilo; cada paso pasa por una cola hacia el flujo SSE.
        4. Si el navegador se desconecta, el criterio de parada detiene el modelo.
        """
        datos = request.get_json(silent=True)
        if not isinstance(datos, dict) or datos.get("consentimiento") != "si":
            return jsonify(errores=["Completa las etapas anteriores y confirma el consentimiento."]), 400
        try:
            entrada = entrada_de_sesion(datos.get("id_corrida"))
        except KeyError:
            return jsonify(errores=["El resultado de la etapa 1 venció o no existe. Vuelve a analizar la referencia."]), 410
        if entrada.identidad is None:
            return jsonify(errores=["Ejecuta primero la etapa 2 para esta referencia: la etapa 3 usa sus 32 vectores."]), 409

        try:
            generador = current_app.extensions["modelos"].obtener("generador", fabrica_generador)
        except ModeloNoDisponible as exc:
            return jsonify(errores=[str(exc)]), 503
        texto = datos.get("texto") if isinstance(datos.get("texto"), str) else ""
        try:
            ids, piezas = generador.tokenizar(texto)
        except ValueError as exc:
            return jsonify(errores=[str(exc)]), 400
        if generador.cerrojo.locked():
            return jsonify(errores=["Ya hay una generación en curso. Espera a que termine."]), 409

        determinista = bool(datos.get("determinista"))
        semilla = datos.get("semilla", 0)
        semilla = semilla if isinstance(semilla, int) and not isinstance(semilla, bool) else 0
        propietario, identificador = session.get("propietario"), entrada.id_corrida
        inicio = generador.evento_inicio(identificador, texto, ids, piezas, determinista, semilla)

        cola = queue.Queue()
        detener = threading.Event()

        def trabajar():
            try:
                resultado = generador.generar(entrada.identidad, texto,
                                              al_paso=lambda p: cola.put(("paso", evento_paso(p))),
                                              detener=detener, determinista=determinista, semilla=semilla)
                try:
                    memoria.anotar(propietario, identificador, tokens_audio=resultado.tokens,
                                   texto_generado=texto.strip())
                except KeyError:
                    pass
                cola.put(("fin", resultado.como_dict()))
            except GeneradorOcupado as exc:
                cola.put(("error", {"errores": [str(exc)]}))
            except Exception:  # el detalle queda en la terminal
                current_app.logger.exception("Error al ejecutar la etapa 3")
                cola.put(("error", {"errores": ["No se pudo completar la etapa 3. Revisa la terminal."]}))
            finally:
                cola.put(None)

        app = current_app._get_current_object()
        hilo = threading.Thread(target=lambda: _con_contexto(app, trabajar), daemon=True)

        def flujo():
            yield formato_sse("inicio", inicio)
            hilo.start()
            try:
                while (item := cola.get()) is not None:
                    yield formato_sse(*item)
            finally:
                # Cierre normal o navegador desconectado: el GPT-2 se detiene en el siguiente paso.
                detener.set()

        return Response(flujo(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @rutas.post("/api/fases/audio")
    def fase_audio():
        """Devuelve el WAV crudo de una fase, calculado en memoria al pedirlo.

        mel               el mel de la etapa 1, de vuelta a sonido (Griffin-Lim)
        tokens_referencia tu voz -> tokens del DVAE -> sonido (requiere reenviar el WAV)
        tokens_texto      los tokens del GPT-2 -> DVAE -> sonido
        """
        if request.form.get("consentimiento") != "si":
            return jsonify(errores=["Confirma el consentimiento."]), 400
        fase = request.form.get("fase")
        if fase not in FASES:
            return jsonify(errores=[f"Fase desconocida: {fase}."]), 400
        try:
            entrada = entrada_de_sesion(request.form.get("id_corrida"))
        except KeyError:
            return jsonify(errores=["El resultado de la etapa 1 venció o no existe. Vuelve a analizar la referencia."]), 410

        try:
            if fase == "mel":
                muestras = audio_desde_mel_referencia(entrada.mel)
            else:
                fases = current_app.extensions["modelos"].obtener("fases", _fabrica_fases)
                if fase == "tokens_texto":
                    if entrada.tokens_audio is None:
                        return jsonify(errores=["Genera primero los tokens en la etapa 3."]), 409
                    muestras = fases.audio_desde_tokens(entrada.tokens_audio)
                else:
                    archivo = request.files.get("audio")
                    if archivo is None:
                        return jsonify(errores=["Esta fase necesita el audio de referencia."]), 400
                    # El servidor no guardó el WAV; se comprueba que el reenviado sea el mismo.
                    audio = cargar_audio(archivo.read())
                    mel = caracteristicas.mel_espectrograma(audio.muestras)
                    if mel.shape != entrada.mel.shape or not np.allclose(mel, entrada.mel, atol=1e-4):
                        return jsonify(errores=["Ese audio no corresponde a la referencia analizada."]), 409
                    muestras = fases.audio_desde_tokens(fases.tokens_de_referencia(audio.muestras))
        except ModeloNoDisponible as exc:
            return jsonify(errores=[str(exc)]), 503
        except (AudioInvalido, ValueError) as exc:
            return jsonify(errores=[str(exc)]), 400

        return Response(a_wav(muestras), mimetype="audio/wav", headers={"Cache-Control": "no-store"})

    return rutas


def _fabrica_fases():
    """El DVAE necesita las mel_stats del checkpoint; las toma del modelo de la etapa 2."""
    identidad = current_app.extensions["modelos"].obtener("identidad", fabrica_identidad)
    return ReconstructorFases(carpeta_modelo(), identidad.normas)


def _con_contexto(app, funcion):
    with app.app_context():
        funcion()
