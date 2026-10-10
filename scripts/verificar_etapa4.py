"""Verificación de las etapas 4 y 5 con el checkpoint oficial de XTTS-v2.

Carga el modelo `Xtts` completo por el camino oficial y comprueba que, con la
misma identidad, el mismo texto y la misma semilla, la etapa 4 de este proyecto
produce el mismo vector de hablante, los mismos latentes y la misma onda.
Necesita unos 5 GB de memoria: hay dos copias del modelo a la vez.
"""

import argparse
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch

from clonvoz.analisis import analizar
from clonvoz.explicacion import Explicador
from clonvoz.generacion import GeneradorTokens
from clonvoz.identidad import CodificadorIdentidad, carpeta_modelo, identidad_desde_mel
from clonvoz.sintesis import Sintetizador


def xtts_oficial(carpeta):
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig

    config = XttsConfig()
    config.load_json(str(carpeta / "config.json"))
    modelo = Xtts.init_from_config(config)
    # Las mismas cuatro clases que admite clonvoz.identidad.leer_checkpoint.
    with torch.serialization.safe_globals([XttsConfig, XttsArgs, XttsAudioConfig, BaseDatasetConfig]):
        modelo.load_checkpoint(config, checkpoint_dir=str(carpeta), eval=True)
    return modelo.to("cpu")


def main():
    parser = argparse.ArgumentParser(description="Verificación de la etapa 4 con checkpoint de XTTS-v2.")
    parser.add_argument("--modelo", type=Path, default=carpeta_modelo())
    parser.add_argument("--texto", default="Hola, esta es una prueba de la etapa cuatro.")
    args = parser.parse_args()

    # Señal artificial explícita, como en las etapas 2 y 3: comprueba el cálculo, no la calidad.
    t = np.arange(int(22050 * 8.1), dtype=np.float32) / 22050
    senal = (0.2 * np.sin(2 * np.pi * (170 * t + 4 * t ** 2))).astype(np.float32)
    archivo = io.BytesIO()
    sf.write(archivo, senal, 22050, format="WAV", subtype="FLOAT")
    etapa1 = analizar(archivo.getvalue(), metodo_f0="yin")
    vectores = identidad_desde_mel(etapa1.mel, CodificadorIdentidad(args.modelo, "cpu"),
                                   etapa1.id_corrida, etapa1.audio.duracion_s).vectores

    generador = GeneradorTokens(args.modelo, "cpu")
    sintetizador = Sintetizador(args.modelo, generador)
    oficial = xtts_oficial(args.modelo)

    # 1. Vector de hablante.
    embedding = sintetizador.embedding_hablante(etapa1.audio.muestras)
    esperado = oficial.get_speaker_embedding(torch.from_numpy(etapa1.audio.muestras)[None], 22050)
    np.testing.assert_allclose(embedding, esperado[0, :, 0].numpy(), atol=1e-5)
    print(f"OK vector de hablante [512] idéntico a Xtts.get_speaker_embedding (norma {np.linalg.norm(embedding):.3f}).")

    # 2. Tokens, latentes y onda, en modo determinista y con semilla.
    for determinista, semilla in ((True, 0), (False, 1234)):
        resultado = generador.generar(vectores, args.texto, determinista=determinista, semilla=semilla)
        inicio = time.perf_counter()
        sintesis = sintetizador.sintetizar(vectores, resultado.texto_ids, resultado.tokens, embedding)
        tardo = time.perf_counter() - inicio

        opciones = {"do_sample": False} if determinista else {"do_sample": True, **generador.muestreo}
        opciones.setdefault("repetition_penalty", generador.muestreo["repetition_penalty"])
        opciones.setdefault("length_penalty", generador.muestreo["length_penalty"])
        torch.manual_seed(semilla)
        salida = oficial.inference(args.texto, "es", torch.from_numpy(vectores)[None], esperado, **opciones)

        modo = "determinista" if determinista else f"muestreo con semilla {semilla}"
        assert salida["gpt_latents"].shape[1] == sintesis.latentes.shape[0], f"Largo distinto en {modo}"
        np.testing.assert_allclose(sintesis.latentes, salida["gpt_latents"][0], atol=2e-3)
        assert salida["wav"].shape == sintesis.onda.shape, f"Onda de largo distinto en {modo}"
        diferencia = float(np.abs(sintesis.onda - salida["wav"]).max())
        assert diferencia < 2e-2 * max(float(np.abs(salida["wav"]).max()), 1e-6), f"Onda distinta en {modo}"
        print(f"OK {modo}: {len(resultado.tokens)} tokens -> latentes {list(sintesis.latentes.shape)} y "
              f"{sintesis.duracion_s:.1f} s de audio iguales a Xtts.inference (diferencia máxima {diferencia:.1e}); "
              f"la etapa 4 tardó {tardo:.1f} s, {tardo / sintesis.duracion_s:.2f} veces el tiempo real.")

    # 3. La pasada con teacher forcing ve lo mismo que la generación.
    pasos = []
    resultado = generador.generar(vectores, args.texto, determinista=True, al_paso=pasos.append)
    explicador = Explicador(generador)
    error = 0.0
    for capa in generador.capas:
        matriz = explicador.atencion(vectores, resultado.texto_ids, resultado.tokens, capa)
        error = max(error, max(float(np.abs(matriz[p.paso, :resultado.prefijo + p.paso] - p.atencion[capa]).max())
                               for p in pasos))
    assert error < 1e-4, "La atención recalculada difiere de la observada en vivo"
    assert pasos[0].zonas.shape == (len(generador.gpt.gpt.h), generador.gpt.gpt.config.n_head, 3)
    assert all(not c.attn._forward_hooks for c in generador.gpt.gpt.h), "Quedaron ganchos puestos"
    print(f"OK interpretabilidad: la atención recalculada coincide con la observada en vivo "
          f"(diferencia máxima {error:.1e}) en {len(pasos)} pasos y {len(generador.capas)} capas.")

    inicio = time.perf_counter()
    efecto = explicador.efecto_causal(vectores, resultado.texto_ids, resultado.tokens)
    print(f"OK intervenciones en {time.perf_counter() - inicio:.1f} s; atención media {efecto['atencion_media']}.")
    for i in efecto["intervenciones"]:
        if i["capas"] == "todas":
            print(f"   sin atención a {i['zona']}: log p cae {i['caida_media']:.3f} nats; cambia el más probable "
                  f"en {100 * i['cambia_el_mas_probable']:.0f} % de los pasos.")


if __name__ == "__main__":
    main()
