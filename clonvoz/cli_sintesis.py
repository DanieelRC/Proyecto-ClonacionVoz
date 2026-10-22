"""Línea de comandos de las etapas 4 y 5: la corrida completa hasta la voz clonada, sin Flask."""

import argparse
import json
import uuid
from pathlib import Path

import numpy as np

from .analisis import AudioRechazado
from .audio import AudioInvalido
from .fases import ReconstructorFases, a_wav
from .generacion import GeneradorTokens
from .grabacion import Grabador, Traductor
from .graficas_sintesis import renderizar_sintesis
from .identidad import CodificadorIdentidad, ModeloNoDisponible, carpeta_modelo
from .pipeline import BLOQUES, ejecutar_corrida
from .sintesis import Sintetizador


class ModelosLocales:
    """Carga cada modelo la primera vez que el pipeline lo pide."""

    def __init__(self, carpeta, dispositivo):
        self.carpeta, self.dispositivo, self.cargados = carpeta, dispositivo, {}

    def _uno(self, nombre, fabrica):
        if nombre not in self.cargados:
            self.cargados[nombre] = fabrica()
        return self.cargados[nombre]

    def identidad(self):
        return self._uno("identidad", lambda: CodificadorIdentidad(self.carpeta, self.dispositivo))

    def generador(self):
        return self._uno("generador", lambda: GeneradorTokens(self.carpeta, self.dispositivo))

    def fases(self):
        return self._uno("fases", lambda: ReconstructorFases(self.carpeta, self.identidad().normas))

    def sintetizador(self):
        return self._uno("sintetizador", lambda: Sintetizador(self.carpeta, self.generador()))


def main(argv=None):
    """Encadena las etapas 1 → 4 con el mismo pipeline que usa el servidor."""
    parser = argparse.ArgumentParser(description="Etapas 4 y 5: voz clonada con XTTS-v2, de principio a fin.")
    parser.add_argument("audio", type=Path, help="Archivo .wav de referencia")
    parser.add_argument("--texto", required=True, help="Texto en español que dirá la voz")
    parser.add_argument("--modelo", type=Path, default=carpeta_modelo(), help="Ruta al checkpoint de XTTS-v2")
    parser.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto", help="Dispositivo de cómputo")
    parser.add_argument("--determinista", action="store_true", help="Elegir siempre el token más probable")
    parser.add_argument("--semilla", type=int, default=0, help="Semilla del muestreo")
    parser.add_argument("--sin-explicacion", action="store_true", help="Omitir las intervenciones causales")
    parser.add_argument("--salida", type=Path, help="Carpeta para voz_clonada.wav, tokens.npy, causal.json y las figuras")
    parser.add_argument("--grabar", type=Path,
                        help="Guardar la corrida completa (.json) para el modo grabado; ponla en grabaciones/")
    parser.add_argument("--consentimiento", action="store_true", help="Confirmar autorización para procesar esta voz")
    args = parser.parse_args(argv)

    if not args.consentimiento:
        parser.error("Confirma la autorización para usar esta voz en clonación académica con --consentimiento.")

    nombres = {identificador: nombre for identificador, nombre, _ in BLOQUES}
    id_corrida = str(uuid.uuid4())
    grabador = Grabador() if args.grabar else None
    traducir = Traductor(args.audio.name, args.texto, id_corrida)
    if grabador:
        grabador("corrida", {"id_corrida": id_corrida, "texto": args.texto.strip(),
                             "determinista": args.determinista, "semilla": args.semilla,
                             "bloques": [{"id": b, "nombre": n, "ruta": r} for b, n, r in BLOQUES]})

    def emitir(tipo, datos):
        if grabador:
            grabador(tipo, traducir(tipo, datos))
        if tipo == "bloque" and datos["estado"] != "activo":
            partes = [f"{nombres[datos['id']]}: {datos['estado']}"]
            if datos.get("ms"):
                partes.append(f"{datos['ms']:.0f} ms")
            if datos.get("forma"):
                partes.append(str(datos["forma"]))
            if datos.get("nota"):
                partes.append(datos["nota"])
            print("  " + " · ".join(partes), flush=True)
        elif tipo == "inicio":
            print("  Fichas BPE:", " ".join(datos["piezas"]), flush=True)
        elif tipo == "explicando":
            print("  Midiendo el efecto de bloquear la atención…", flush=True)

    try:
        corrida = ejecutar_corrida(args.audio, args.texto, ModelosLocales(args.modelo, args.dispositivo),
                                   emitir, determinista=args.determinista, semilla=args.semilla,
                                   explicar=not args.sin_explicacion, id_corrida=id_corrida)
    except (ModeloNoDisponible, AudioInvalido, AudioRechazado, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrumpida.")
        return 130

    if grabador:
        grabador("terminada", {"motivo": "completa"})
        args.grabar.parent.mkdir(parents=True, exist_ok=True)
        grabador.guardar(args.grabar)
        print(f"Corrida grabada en {args.grabar}: ábrela en la página con el modo Grabado.")

    sintesis = corrida.sintesis
    print(f"\nVoz clonada: {sintesis.duracion_s:.1f} s a {sintesis.sr} Hz, "
          f"{len(corrida.generacion.tokens)} tokens.")
    if corrida.causal:
        print(f"Atención media: {corrida.causal['atencion_media']}")
        for i in corrida.causal["intervenciones"]:
            if i["capas"] == "todas":
                print(f"Sin atención a {i['zona']}: log p del token elegido cae {i['caida_media']:.3f} nats; "
                      f"cambia el más probable en {100 * i['cambia_el_mas_probable']:.0f} % de los pasos.")

    if args.salida:
        args.salida.mkdir(parents=True, exist_ok=True)
        (args.salida / "voz_clonada.wav").write_bytes(a_wav(sintesis.onda, sr=sintesis.sr))
        np.save(args.salida / "tokens.npy", corrida.generacion.tokens)
        np.save(args.salida / "latentes.npy", sintesis.latentes)
        resumen = {"id_corrida": corrida.analisis.id_corrida, "texto": args.texto.strip(),
                   "generacion": corrida.generacion.como_dict(), "sintesis": sintesis.como_dict()}
        (args.salida / "corrida.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
        if corrida.causal:
            (args.salida / "causal.json").write_text(json.dumps(corrida.causal, ensure_ascii=False), encoding="utf-8")
        for nombre, temas in renderizar_sintesis(sintesis, corrida.analisis.mel).items():
            for tema, png in temas.items():
                (args.salida / f"{nombre}_{tema}.png").write_bytes(png)
        print(f"Exportado en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
