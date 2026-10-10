"""Interpretabilidad del GPT-2 más allá del mapa de atención en vivo.

La atención dice a dónde miró el modelo, no qué causó su decisión. Aquí se añade
lo que el mapa en vivo no puede dar:

- la atención de cualquier capa y cualquier cabeza, sin promediar y con una sola
  escala para toda la matriz;
- intervenciones: se anula una parte de la atención y se mide cuánto cambia la
  probabilidad del token que el modelo había elegido.

Todo sale de una pasada sobre la secuencia ya generada (teacher forcing) con el
mismo `gpt_inference` oficial. `head_mask` es el argumento estándar de GPT-2 que
multiplica los pesos de atención después del softmax; con él se hacen las
intervenciones, sin tocar el modelo.
"""

from __future__ import annotations

import base64

import numpy as np

from .generacion import GeneradorOcupado

ZONAS = ("voz", "texto", "audio")
TRAMOS = ("primer tercio", "segundo tercio", "último tercio")


def matriz_cuantizada(matriz) -> dict:
    """Matriz de atención -> 8 bits en base64 con un único máximo para toda ella."""
    maximo = float(matriz.max()) if matriz.size else 0.0
    escala = matriz / maximo if maximo > 0 else np.zeros_like(matriz)
    q = np.clip(np.rint(escala * 255), 0, 255).astype(np.uint8)
    return {"max": maximo, "filas": int(matriz.shape[0]), "columnas": int(matriz.shape[1]),
            "q": base64.b64encode(q.tobytes()).decode("ascii")}


class Explicador:
    """Atención completa e intervenciones sobre una secuencia ya generada."""

    def __init__(self, generador):
        self.generador = generador
        self.torch = generador.torch
        self.gpt = generador.gpt
        self.capas = len(self.gpt.gpt.h)
        self.cabezas = self.gpt.gpt.config.n_head

    # ------------------------------------------------------------------ pasadas

    def _preparar(self, vectores, ids, tokens):
        """Arma la secuencia [voz | texto | inicio, tokens[:-1]] que el modelo vio al generar."""
        torch = self.torch
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.ndim != 1 or tokens.size == 0:
            raise ValueError("No hay tokens generados que explicar.")
        vectores = np.asarray(vectores, dtype=np.float32)
        if vectores.shape != (32, self.gpt.model_dim):
            raise ValueError(f"Se esperaba una identidad de forma [32, {self.gpt.model_dim}].")
        disp = self.generador.dispositivo
        cond = torch.from_numpy(vectores)[None].to(disp)
        texto = torch.IntTensor([int(i) for i in ids])[None].to(disp)
        entrada = self.gpt.compute_embeddings(cond, texto)
        codigos = torch.from_numpy(tokens[:-1])[None].to(disp)
        return torch.cat([entrada, codigos], dim=1), tokens, entrada.shape[1]

    def _logaritmos(self, entrada, prefijo, mascara=None):
        """log p de cada paso, [pasos, vocabulario], con una máscara de cabezas opcional."""
        salida = self.gpt.gpt_inference(input_ids=entrada, head_mask=mascara, use_cache=False,
                                        return_dict=True)
        return self.torch.log_softmax(salida.logits[0, prefijo - 1:].float(), dim=-1)

    def _con_cerrojo(self, funcion):
        if not self.generador.cerrojo.acquire(blocking=False):
            raise GeneradorOcupado("El GPT-2 está ocupado. Espera a que termine.")
        try:
            with self.torch.inference_mode():
                return funcion()
        finally:
            self.generador.cerrojo.release()

    # ----------------------------------------------------------------- atención

    def atencion(self, vectores, ids, tokens, capa, cabeza=None) -> np.ndarray:
        """Atención de una capa [pasos, secuencia]; una cabeza o el promedio si `cabeza` es None.

        Fila k: lo que miró el modelo al elegir el token k. Es la misma fila que el
        gancho ve en vivo, pero disponible para las 30 capas y las 16 cabezas.
        """
        if not 0 <= capa < self.capas:
            raise ValueError(f"La capa debe estar entre 1 y {self.capas}.")
        if cabeza is not None and not 0 <= cabeza < self.cabezas:
            raise ValueError(f"La cabeza debe estar entre 1 y {self.cabezas}.")

        def calcular():
            entrada, _, prefijo = self._preparar(vectores, ids, tokens)
            captura = {}

            def gancho(_modulo, _entrada, salida):
                pesos = salida[1][0, :, prefijo - 1:, :].float()
                captura["a"] = (pesos.mean(dim=0) if cabeza is None else pesos[cabeza]).cpu().numpy()

            puesto = self.gpt.gpt.h[capa].attn.register_forward_hook(gancho)
            try:
                self.gpt.gpt_inference(input_ids=entrada, use_cache=False, return_dict=True)
            finally:
                puesto.remove()
            return captura["a"]

        return self._con_cerrojo(calcular)

    # ------------------------------------------------------------ intervenciones

    def _mascara_zona(self, largo, prefijo, zona, capas):
        """Ceros donde una posición de audio mira a `zona`, solo en las capas indicadas."""
        torch = self.torch
        inicio, fin = (0, 32) if zona == "voz" else (32, prefijo - 1)
        corte = torch.ones(largo, largo, device=self.generador.dispositivo)
        corte[prefijo - 1:, inicio:fin] = 0
        if capas is None:
            return corte.expand(self.capas, 1, 1, largo, largo)
        activa = torch.zeros(self.capas, 1, 1, 1, 1, device=corte.device)
        activa[list(capas)] = 1
        return 1 - activa * (1 - corte)

    def efecto_causal(self, vectores, ids, tokens, por_tramos=True) -> dict:
        """Cuánto cambia la decisión al impedir que el audio mire a la voz o al texto.

        1. Pasada base: log p del token elegido en cada paso y la atención real a cada zona.
        2. Por cada intervención, otra pasada con esos pesos de atención en cero.
        3. Se comparan: caída de log p del token elegido, divergencia KL y si cambia
           el token más probable.

        Es una intervención real sobre el cálculo, pero saca al modelo de su régimen
        normal: mide dependencia, no "importancia" en un sentido único.
        """
        def calcular():
            torch = self.torch
            entrada, elegidos, prefijo = self._preparar(vectores, ids, tokens)
            largo, pasos = entrada.shape[1], len(elegidos)
            indice = torch.from_numpy(elegidos).to(entrada.device)[:, None]

            # Base, con ganchos en todas las capas para la atención media a cada zona.
            por_capa = []

            def gancho(_modulo, _entrada, salida):
                pesos = salida[1][0, :, prefijo - 1:, :].float().mean(dim=0)
                por_capa.append(torch.stack((pesos[:, :32].sum(dim=1), pesos[:, 32:prefijo - 1].sum(dim=1),
                                             pesos[:, prefijo - 1:].sum(dim=1)), dim=1).cpu())

            ganchos = [capa.attn.register_forward_hook(gancho) for capa in self.gpt.gpt.h]
            try:
                base = self._logaritmos(entrada, prefijo)
            finally:
                for g in ganchos:
                    g.remove()
            por_capa = torch.stack(por_capa)  # [capas, pasos, 3]
            masas = por_capa.mean(dim=0)
            base_elegido = base.gather(1, indice)[:, 0]

            tramos = [None]
            if por_tramos and self.capas >= 3:
                tramos += [[int(c) for c in tramo] for tramo in np.array_split(np.arange(self.capas), 3)]

            intervenciones = []
            for zona in ("voz", "texto"):
                for n, capas in enumerate(tramos):
                    con = self._logaritmos(entrada, prefijo, self._mascara_zona(largo, prefijo, zona, capas))
                    caida = base_elegido - con.gather(1, indice)[:, 0]
                    kl = (base.exp() * (base - con)).sum(dim=1)
                    cambia = base.argmax(dim=1) != con.argmax(dim=1)
                    de_esas_capas = por_capa if capas is None else por_capa[capas]
                    intervenciones.append({
                        "zona": zona,
                        "atencion_recibida": round(float(de_esas_capas[:, :, ZONAS.index(zona)].mean()), 4),
                        "capas": "todas" if capas is None else TRAMOS[n - 1],
                        "rango": [1, self.capas] if capas is None else [capas[0] + 1, capas[-1] + 1],
                        "caida_media": round(float(caida.mean()), 6),
                        "kl_media": round(float(kl.mean()), 6),
                        "cambia_el_mas_probable": round(float(cambia.float().mean()), 4),
                        "caida": [round(float(v), 3) for v in caida.cpu()],
                        "kl": [round(float(v), 3) for v in kl.cpu()],
                    })
            return {"pasos": pasos, "capas": self.capas, "cabezas": self.cabezas,
                    "log_p_base": [round(float(v), 3) for v in base_elegido.cpu()],
                    "atencion": {zona: [round(float(v), 4) for v in masas[:, i]]
                                 for i, zona in enumerate(ZONAS)},
                    "atencion_media": {zona: round(float(masas[:, i].mean()), 4)
                                       for i, zona in enumerate(ZONAS)},
                    "intervenciones": intervenciones}

        return self._con_cerrojo(calcular)

    def efecto_cabezas(self, vectores, ids, tokens, capa) -> dict:
        """Apaga una por una las cabezas de una capa y mide el cambio en la decisión."""
        if not 0 <= capa < self.capas:
            raise ValueError(f"La capa debe estar entre 1 y {self.capas}.")

        def calcular():
            torch = self.torch
            entrada, elegidos, prefijo = self._preparar(vectores, ids, tokens)
            indice = torch.from_numpy(elegidos).to(entrada.device)[:, None]
            base = self._logaritmos(entrada, prefijo)
            base_elegido = base.gather(1, indice)[:, 0]
            cabezas = []
            for cabeza in range(self.cabezas):
                mascara = torch.ones(self.capas, 1, self.cabezas, 1, 1, device=entrada.device)
                mascara[capa, 0, cabeza] = 0
                con = self._logaritmos(entrada, prefijo, mascara)
                caida = base_elegido - con.gather(1, indice)[:, 0]
                kl = (base.exp() * (base - con)).sum(dim=1)
                cabezas.append({"cabeza": cabeza + 1, "caida_media": round(float(caida.mean()), 6),
                                "kl_media": round(float(kl.mean()), 6)})
            return {"capa": capa + 1, "cabezas": cabezas}

        return self._con_cerrojo(calcular)
