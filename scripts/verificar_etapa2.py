"""Verificación con checkpoint REAL, sin construir el generador GPT-2.

Compara el resultado instrumentado con un recorrido sin hooks de los mismos
módulos oficiales. Usa una señal artificial explícita para no necesitar voces.
"""

import argparse
import sys
import io
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import soundfile as sf

from clonvoz import caracteristicas
from clonvoz.identidad import CodificadorIdentidad, carpeta_modelo


def main():
    """Verifica tres propiedades distintas con el checkpoint descargado.

    1. Capturar la atención no cambia los vectores; las probabilidades de audio
       y consultas juntas suman uno, y el hook se retira después de usarlo.
    2. Nuestro log-mel normalizado coincide con wav_to_mel_cloning oficial.
    3. El mel completo pasa directamente de etapa 1 a get_style_emb oficial.

    Las señales son tonos generados por una fórmula, no voces. Estas pruebas
    verifican compatibilidad numérica, no reconocimiento de hablantes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", type=Path, default=carpeta_modelo())
    args = parser.parse_args()
    modelo = CodificadorIdentidad(args.modelo, "cpu")
    t = np.arange(22050 * 3, dtype=np.float32) / 22050
    # Tres segundos de seno a 170 Hz: una entrada conocida y repetible.
    mel = caracteristicas.mel_espectrograma(0.2 * np.sin(2 * np.pi * 170 * t))
    capas = []
    vectores, atencion, propias = modelo.procesar_mel(mel, diagnostico=capas)
    assert len(capas) == 2
    for capa, modulo in zip(capas, modelo.perceiver.layers):
        np.testing.assert_allclose(capa.sum(axis=1), 1, atol=1e-6)
        assert not modulo[0].attend.attn_dropout._forward_hooks
        assert np.isfinite(capa).all() and (capa >= 0).all()
        print(f"Atencion al audio: maximo={capa[:, 32:].max():.3e}, "
              f"masa media={capa[:, 32:].sum(axis=1).mean():.3e}")
    with torch.inference_mode():
        # 1. Repetir el recorrido usando los mismos módulos, ahora sin el hook.
        entrada = torch.from_numpy(mel / modelo.normas[:, None]).unsqueeze(0)
        referencia = modelo.perceiver(modelo.encoder(entrada).transpose(1, 2))[0].numpy()
    np.testing.assert_allclose(vectores, referencia, rtol=1e-5, atol=1e-6)
    # allclose permite pequeñas diferencias de redondeo de punto flotante.
    np.testing.assert_allclose(atencion.sum(axis=1) + propias.sum(axis=1), 1, atol=1e-6)
    assert not modelo.perceiver.layers[-1][0].attend.attn_dropout._forward_hooks
    assert atencion.shape == (32, mel.shape[1])
    assert vectores.shape == (32, 1024)
    # Comparar también con get_style_emb oficial, sin construir GPT-2 completo.
    # Este método solo necesita sus dos módulos de condicionamiento y la bandera.
    from TTS.tts.layers.xtts.gpt import GPT
    from TTS.tts.models.xtts import wav_to_mel_cloning
    from clonvoz.analisis import analizar
    from clonvoz.identidad import identidad_desde_mel

    solo_codificacion = SimpleNamespace(conditioning_encoder=modelo.encoder,
                                        conditioning_perceiver=modelo.perceiver,
                                        use_perceiver_resampler=True)
    t_largo = np.arange(int(22050 * 8.1), dtype=np.float32) / 22050
    senal = (0.2 * np.sin(2 * np.pi * (170 * t_largo + 4 * t_largo ** 2))).astype(np.float32)
    archivo = io.BytesIO()
    sf.write(archivo, senal, 22050, format="WAV", subtype="FLOAT")
    etapa1 = analizar(archivo.getvalue(), metodo_f0="yin")
    # A partir de aquí, incluso intentar recalcular un mel hace fallar la prueba.
    from unittest.mock import patch
    with patch("clonvoz.caracteristicas.mel_espectrograma", side_effect=AssertionError("Mel recalculado")):
        nuestro = identidad_desde_mel(etapa1.mel, modelo, etapa1.id_corrida,
                                      etapa1.audio.duracion_s, etapa1.validacion.advertencias)
    mel_oficial = wav_to_mel_cloning(torch.from_numpy(senal).unsqueeze(0),
                                    mel_norms=torch.from_numpy(modelo.normas),
                                    n_fft=2048, hop_length=256, win_length=1024)
    np.testing.assert_allclose(etapa1.mel / modelo.normas[:, None], mel_oficial[0].numpy(), atol=1e-5)
    with torch.inference_mode():
        oficial = GPT.get_style_emb(solo_codificacion, mel_oficial).transpose(1, 2)[0].numpy()
    np.testing.assert_allclose(nuestro.vectores, oficial, atol=1e-5, rtol=1e-5)
    assert nuestro.id_corrida == etapa1.id_corrida
    assert nuestro.atencion.shape == (32, etapa1.mel.shape[1])
    print("OK: mel completo de etapa 1 reutilizado sin recalcular; mismo identificador.")
    print("OK: vectores iguales a get_style_emb oficial; atención y retirada de hooks verificadas.")
    print("Señal artificial: prueba numérica, no una evaluación de voces humanas.")


if __name__ == "__main__":
    main()
