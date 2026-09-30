"""Verificación de la etapa 3 con el checkpoint oficial de XTTS-v2."""

import argparse
import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch

from clonvoz import config
from clonvoz.analisis import analizar
from clonvoz.fases import MUESTRAS_POR_CODIGO, ReconstructorFases
from clonvoz.generacion import GeneradorTokens
from clonvoz.identidad import CodificadorIdentidad, carpeta_modelo, identidad_desde_mel


class Capturado(Exception):
    """Detiene Xtts.inference justo al llamar al GPT, con sus argumentos."""


def argumentos_oficiales(generador, vectores, texto, **opciones):
    """Ejecuta `Xtts.inference` oficial hasta `gpt.generate` y devuelve lo que le pasaría."""
    from TTS.tts.models.xtts import Xtts

    def capturar(**kwargs):
        raise Capturado(kwargs)

    falso = SimpleNamespace(device="cpu", tokenizer=generador.tokenizador, gpt_batch_size=1,
                            args=SimpleNamespace(gpt_max_text_tokens=generador.gpt.max_text_tokens),
                            gpt=SimpleNamespace(generate=capturar))
    try:
        Xtts.inference(falso, texto, "es", torch.from_numpy(vectores)[None], torch.zeros(1, 512, 1),
                       **opciones)
    except Capturado as captura:
        return captura.args[0]
    raise AssertionError("Xtts.inference no llegó a gpt.generate")


def main():
    """Compara la etapa 3 instrumentada con el camino oficial de XTTS-v2."""
    parser = argparse.ArgumentParser(description="Verificación de la etapa 3 con checkpoint de XTTS-v2.")
    parser.add_argument("--modelo", type=Path, default=carpeta_modelo())
    parser.add_argument("--texto", default="Hola, esta es una prueba de la etapa tres.")
    args = parser.parse_args()

    # Una señal artificial explícita, como en la etapa 2: comprueba el cálculo, no la calidad.
    t = np.arange(int(22050 * 8.1), dtype=np.float32) / 22050
    senal = (0.2 * np.sin(2 * np.pi * (170 * t + 4 * t ** 2))).astype(np.float32)
    archivo = io.BytesIO()
    sf.write(archivo, senal, 22050, format="WAV", subtype="FLOAT")
    etapa1 = analizar(archivo.getvalue(), metodo_f0="yin")
    codificador = CodificadorIdentidad(args.modelo, "cpu")
    vectores = identidad_desde_mel(etapa1.mel, codificador, etapa1.id_corrida,
                                   etapa1.audio.duracion_s).vectores

    inicio = time.perf_counter()
    generador = GeneradorTokens(args.modelo, "cpu")
    print(f"GPT-2 cargado en {time.perf_counter() - inicio:.1f} s; capas observadas "
          f"{[c + 1 for c in generador.capas]} de {len(generador.gpt.gpt.h)}; tope {generador.tope}.")

    ids, piezas = generador.tokenizar(args.texto)
    print("Fichas:", " ".join(piezas))
    capas = generador.gpt.gpt.h

    for determinista, semilla in ((True, 0), (False, 1234)):
        opciones = {"do_sample": False} if determinista else dict(generador.muestreo)
        oficial = argumentos_oficiales(generador, vectores, args.texto, **opciones)
        assert oficial["text_inputs"][0].tolist() == ids, "La tokenización difiere de Xtts.inference"
        if not determinista:
            for clave, valor in generador.muestreo.items():
                assert oficial[clave] == valor, clave

        resultado = generador.generar(vectores, args.texto, determinista=determinista, semilla=semilla)
        assert all(not c.attn._forward_hooks for c in capas), "Quedaron ganchos puestos"
        for paso in resultado.pasos:
            for fila in paso.atencion.values():
                assert fila.shape == (resultado.prefijo + paso.paso,)
                np.testing.assert_allclose(fila.sum(), 1, atol=1e-4)

        torch.manual_seed(semilla)
        with torch.inference_mode():
            referencia = generador.gpt.generate(**oficial)[0].numpy()
        modo = "determinista" if determinista else f"muestreo con semilla {semilla}"
        assert np.array_equal(resultado.tokens, referencia), f"Tokens distintos en {modo}"
        print(f"OK {modo}: {len(referencia)} tokens idénticos a Xtts.inference ({resultado.motivo}), "
              f"{resultado.ms_por_token:.1f} ms por token con ganchos.")

    # Eager (necesario para leer la atención) frente a SDPA, sin ganchos.
    oficial = argumentos_oficiales(generador, vectores, args.texto, do_sample=False)
    tiempos, salidas = {}, {}
    for modo in ("eager", "sdpa"):
        generador.gpt.gpt.set_attn_implementation(modo)
        generador.gpt.gpt._attn_implementation = modo
        inicio = time.perf_counter()
        with torch.inference_mode():
            salidas[modo] = generador.gpt.generate(**oficial)[0].numpy()
        tiempos[modo] = (time.perf_counter() - inicio) * 1000 / len(salidas[modo])
    generador.gpt.gpt.set_attn_implementation("eager")
    generador.gpt.gpt._attn_implementation = "eager"
    iguales = np.array_equal(salidas["eager"], salidas["sdpa"])
    print(f"Eager {tiempos['eager']:.1f} ms/token, SDPA {tiempos['sdpa']:.1f} ms/token; "
          f"tokens {'idénticos' if iguales else 'distintos (solo por redondeo numérico)'}.")

    try:
        fases = ReconstructorFases(args.modelo, codificador.normas)
    except Exception as exc:  # sin dvae.pth las fases 3 y 4 no se pueden comprobar
        print(f"Fases 3 y 4 sin comprobar: {exc}")
        return
    codigos = fases.tokens_de_referencia(senal)
    reconstruido = fases.mel_desde_tokens(codigos)
    with torch.inference_mode():
        original = fases.mel_dvae(torch.from_numpy(senal)[None])[0].numpy() * fases.normas[:, None]
    cuadros = min(original.shape[1], reconstruido.shape[1])
    error = float(np.abs(original[:, :cuadros] - reconstruido[:, :cuadros]).mean())
    audio = fases.audio_desde_tokens(codigos)
    assert reconstruido.shape[1] == 4 * len(codigos)
    # La STFT centrada entrega un salto menos: 256 · (4N − 1) muestras.
    assert len(audio) == len(codigos) * MUESTRAS_POR_CODIGO - config.HOP_LENGTH
    print(f"OK DVAE: {len(codigos)} códigos para {len(senal) / 22050:.1f} s; error medio del "
          f"log-mel de ida y vuelta {error:.3f} (media del log-mel {np.abs(original).mean():.3f}).")


if __name__ == "__main__":
    main()
