"""Rutas de la etapa 4: la voz clonada y su contraste sin vector de hablante."""

import numpy as np
from flask import Blueprint, Response, current_app, jsonify, request, session

from .. import caracteristicas
from ..audio import AudioInvalido, cargar_audio
from ..fases import a_wav
from ..generacion import GeneradorOcupado
from ..grabacion import respuesta_sintesis
from ..identidad import ModeloNoDisponible, carpeta_modelo
from ..sintesis import Sintetizador
from .generacion import _fabrica_fases, fabrica_generador


def fabrica_sintetizador():
    """El HiFi-GAN comparte el GPT-2 ya cargado por la etapa 3."""
    generador = current_app.extensions["modelos"].obtener("generador", fabrica_generador)
    return Sintetizador(carpeta_modelo(), generador)


def fases_o_nada():
    """El DVAE es la rama didáctica: si falta dvae.pth la etapa 4 sigue sin su mel."""
    try:
        return current_app.extensions["modelos"].obtener("fases", _fabrica_fases)
    except ModeloNoDisponible:
        return None


def crear_rutas_sintesis(memoria):
    """Crea el blueprint de la etapa 4 sobre la memoria compartida con las etapas anteriores."""
    rutas = Blueprint("sintesis", __name__)

    def entrada_de_sesion(identificador):
        if not isinstance(identificador, str) or not identificador:
            raise KeyError(identificador)
        return memoria.obtener(session.get("propietario"), identificador)

    @rutas.post("/api/sintetizar")
    def sintetizar():
        """Convierte los tokens de la etapa 3 en voz.

        1. Recuperar identidad, texto y tokens guardados por las etapas 2 y 3.
        2. Vector de hablante: necesita el WAV, que el servidor no guardó. Se reenvía
           una vez y se comprueba que sea el de la referencia analizada.
        3. Latentes del GPT-2 y HiFi-GAN; el DVAE aporta el mel de los tokens.
        """
        if request.form.get("consentimiento") != "si":
            return jsonify(errores=["Confirma el consentimiento."]), 400
        try:
            entrada = entrada_de_sesion(request.form.get("id_corrida"))
        except KeyError:
            return jsonify(errores=["El resultado de la etapa 1 venció o no existe. Vuelve a analizar la referencia."]), 410
        if entrada.identidad is None or entrada.tokens_audio is None or entrada.texto_ids is None:
            return jsonify(errores=["Genera primero los tokens en la etapa 3: la etapa 4 los convierte en voz."]), 409
        if not (np.asarray(entrada.tokens_audio) < 1024).any():
            return jsonify(errores=["La generación no produjo tokens de audio que sintetizar."]), 409

        try:
            sintetizador = current_app.extensions["modelos"].obtener("sintetizador", fabrica_sintetizador)
            embedding = entrada.embedding_hablante
            if embedding is None:
                archivo = request.files.get("audio")
                if archivo is None:
                    return jsonify(errores=["La etapa 4 necesita el audio de referencia para el vector de hablante."]), 400
                audio = cargar_audio(archivo.read())
                mel = caracteristicas.mel_espectrograma(audio.muestras)
                if mel.shape != entrada.mel.shape or not np.allclose(mel, entrada.mel, atol=1e-4):
                    return jsonify(errores=["Ese audio no corresponde a la referencia analizada."]), 409
                embedding = sintetizador.embedding_hablante(audio.muestras)
            sintesis = sintetizador.sintetizar(entrada.identidad, entrada.texto_ids, entrada.tokens_audio,
                                               embedding, fases_o_nada())
        except ModeloNoDisponible as exc:
            return jsonify(errores=[str(exc)]), 503
        except GeneradorOcupado as exc:
            return jsonify(errores=[str(exc)]), 409
        except (AudioInvalido, ValueError) as exc:
            return jsonify(errores=[str(exc)]), 400
        except Exception:
            current_app.logger.exception("Error al ejecutar la etapa 4")
            return jsonify(errores=["No se pudo completar la etapa 4. Revisa la terminal."]), 500

        try:
            memoria.anotar(session.get("propietario"), entrada.id_corrida,
                           embedding_hablante=embedding, latentes=sintesis.latentes)
        except KeyError:
            pass
        datos = respuesta_sintesis(sintesis, entrada.mel)
        datos["id_corrida"] = entrada.id_corrida
        datos["texto"] = entrada.texto_generado
        return jsonify(datos)

    @rutas.post("/api/sintetizar/contraste")
    def contraste():
        """Los mismos latentes por el vocoder con el vector de hablante en cero.

        Es una intervención audible: lo único que cambia es ese vector, así que la
        diferencia que se oye es lo que aporta.
        """
        datos = request.get_json(silent=True)
        if not isinstance(datos, dict) or datos.get("consentimiento") != "si":
            return jsonify(errores=["Confirma el consentimiento."]), 400
        try:
            entrada = entrada_de_sesion(datos.get("id_corrida"))
        except KeyError:
            return jsonify(errores=["El resultado de la etapa 1 venció o no existe. Vuelve a analizar la referencia."]), 410
        if entrada.latentes is None or entrada.embedding_hablante is None:
            return jsonify(errores=["Ejecuta primero la etapa 4."]), 409
        try:
            sintetizador = current_app.extensions["modelos"].obtener("sintetizador", fabrica_sintetizador)
            onda = sintetizador.onda(entrada.latentes, np.zeros_like(entrada.embedding_hablante))
        except ModeloNoDisponible as exc:
            return jsonify(errores=[str(exc)]), 503
        return Response(a_wav(onda, sr=sintetizador.sr), mimetype="audio/wav",
                        headers={"Cache-Control": "no-store"})

    return rutas
