"""Línea de comandos de la etapa 3: tokens de audio con el GPT-2 de XTTS-v2, sin Flask."""

import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .analisis import AudioRechazado, analizar
from .audio import AudioInvalido
from .generacion import GeneradorTokens, evento_paso
from .identidad import CodificadorIdentidad, ModeloNoDisponible, carpeta_modelo, identidad_desde_mel

FORMATO_CORRIDA = "clonvoz-corrida-etapa3"
"""El mismo formato que descarga la página: se reproduce con Reproducir corrida."""


def main(argv=None):
    """Encadena las etapas 1 → 2 → 3 e imprime los tokens conforme salen."""
    parser = argparse.ArgumentParser(description="Etapa 3: GPT-2 de XTTS-v2, tokens de audio sin síntesis.")
    parser.add_argument("audio", type=Path, help="Archivo .wav de referencia")
    parser.add_argument("--texto", required=True, help="Texto en español que dirá la voz")
    parser.add_argument("--modelo", type=Path, default=carpeta_modelo(), help="Ruta al checkpoint de XTTS-v2")
    parser.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto", help="Dispositivo de cómputo")
    parser.add_argument("--determinista", action="store_true", help="Elegir siempre el token más probable")
    parser.add_argument("--semilla", type=int, default=0, help="Semilla del muestreo")
    parser.add_argument("--salida", type=Path, help="Carpeta para tokens.npy, corrida.json y los WAV de las fases")
    parser.add_argument("--consentimiento", action="store_true", help="Confirmar autorización para procesar esta voz")
    args = parser.parse_args(argv)

    if not args.consentimiento:
        parser.error("Confirma la autorización para usar esta voz en clonación académica con --consentimiento.")

    try:
        etapa1 = analizar(args.audio)
        codificador = CodificadorIdentidad(args.modelo, args.dispositivo)
        identidad = identidad_desde_mel(etapa1.mel, codificador, etapa1.id_corrida,
                                        etapa1.audio.duracion_s, etapa1.validacion.advertencias)
        generador = GeneradorTokens(args.modelo, args.dispositivo)
        ids, piezas = generador.tokenizar(args.texto)
    except (ModeloNoDisponible, AudioInvalido, AudioRechazado, ValueError) as exc:
        print(f"Error: {exc}")
        return 2

    inicio = generador.evento_inicio(etapa1.id_corrida, args.texto, ids, piezas,
                                     args.determinista, args.semilla)
    print("Fichas BPE:", " ".join(piezas))
    print(f"Secuencia inicial: {inicio['prefijo']} posiciones; capas observadas {inicio['capas']}; "
          f"tope {generador.tope} tokens.")

    reloj = time.perf_counter()
    eventos = [{"tipo": "inicio", "t": 0, "datos": inicio}]

    def al_paso(paso):
        eventos.append({"tipo": "paso", "t": round((time.perf_counter() - reloj) * 1000),
                        "datos": evento_paso(paso)})
        print(paso.token, end=" ", flush=True)

    try:
        resultado = generador.generar(identidad.vectores, args.texto, al_paso=al_paso,
                                      determinista=args.determinista, semilla=args.semilla)
    except KeyboardInterrupt:
        print("\nInterrumpida.")
        return 130
    fin = resultado.como_dict()
    eventos.append({"tipo": "fin", "t": round((time.perf_counter() - reloj) * 1000), "datos": fin})
    print(f"\n\n{fin['total_tokens']} tokens ({fin['motivo']}), {fin['ms_por_token']} ms por token, "
          f"{fin['ms_total'] / 1000:.1f} s en total.")

    if args.salida:
        args.salida.mkdir(parents=True, exist_ok=True)
        np.save(args.salida / "tokens.npy", resultado.tokens)
        audios = exportar_fases(args.salida, args.modelo, codificador.normas, etapa1, resultado.tokens)
        corrida = {"formato": FORMATO_CORRIDA, "version": 1,
                   "guardada": datetime.now(timezone.utc).isoformat(), "eventos": eventos,
                   "audios": audios}
        (args.salida / "corrida.json").write_text(json.dumps(corrida, ensure_ascii=False), encoding="utf-8")
        print(f"Exportado en {args.salida}")
    return 0


def exportar_fases(carpeta, modelo, normas, etapa1, tokens):
    """Escribe los WAV de las fases 2 a 4 y los devuelve como data URL para la corrida."""
    from .fases import ReconstructorFases, a_wav, audio_desde_mel_referencia

    wavs = {"mel": a_wav(audio_desde_mel_referencia(etapa1.mel))}
    try:
        fases = ReconstructorFases(modelo, normas)
        wavs["tokens_referencia"] = a_wav(fases.audio_desde_tokens(fases.tokens_de_referencia(etapa1.audio.muestras)))
        wavs["tokens_texto"] = a_wav(fases.audio_desde_tokens(tokens))
    except (ModeloNoDisponible, ValueError) as exc:
        print(f"Fases 3 y 4 omitidas: {exc}", file=sys.stderr)

    nombres = {"mel": "fase2_mel.wav", "tokens_referencia": "fase3_tokens_referencia.wav",
               "tokens_texto": "fase4_tokens_texto.wav"}
    for fase, contenido in wavs.items():
        (carpeta / nombres[fase]).write_bytes(contenido)
    return {fase: "data:audio/wav;base64," + base64.b64encode(contenido).decode("ascii")
            for fase, contenido in wavs.items()}


if __name__ == "__main__":
    raise SystemExit(main())
