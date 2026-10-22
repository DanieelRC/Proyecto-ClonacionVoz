/* Diagrama de arquitectura de XTTS-v2 en SVG, animado con los eventos de la corrida.
 * Los nodos son los bloques que emite clonvoz/pipeline.py; cuando uno se activa,
 * un punto recorre las conexiones que llegan a él, y durante la generación otro
 * punto da la vuelta GPT-2 → tokens → GPT-2 con cada token.
 */
(function () {
  "use strict";
  const SVG = "http://www.w3.org/2000/svg";
  const contenedor = document.getElementById("arquitectura");
  const detalle = document.getElementById("detalle-bloque");
  const reducido = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const color = (nombre) => getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();

  const RUTAS = { audio: "--ruta-audio", texto: "--ruta-texto", generacion: "--ruta-generacion", salida: "--ruta-salida" };

  // Grupos de fondo: qué se dice, quién lo dice, cómo se genera y cómo suena.
  const GRUPOS = [
    { x: 270, y: 18, w: 220, h: 178, ruta: "texto", titulo: "Contenido: qué se dice" },
    { x: 640, y: 18, w: 440, h: 320, ruta: "audio", titulo: "Identidad: quién lo dice" },
    { x: 270, y: 262, w: 220, h: 300, ruta: "generacion", titulo: "Generación autorregresiva" },
    { x: 700, y: 418, w: 280, h: 200, ruta: "salida", titulo: "Decodificación" },
  ];

  // x, y: centro. forma: caja, pastilla o punteada.
  const NODOS = [
    { id: "entrada", x: 380, y: 72, w: 170, h: 38, forma: "pastilla", ruta: "texto", nombre: "Texto de entrada",
      ayuda: "La frase que dirá la voz clonada." },
    { id: "texto", x: 380, y: 148, w: 190, h: 54, ruta: "texto", nombre: "Tokenizador BPE",
      ayuda: "Parte el texto en fichas de un vocabulario de 6681." },
    { id: "audio", x: 860, y: 72, w: 190, h: 44, forma: "pastilla", ruta: "audio", nombre: "Audio de referencia",
      ayuda: "Tu voz: se carga, se valida y se remuestrea." },
    { id: "mel", x: 760, y: 186, w: 176, h: 54, ruta: "audio", nombre: "Mel-espectrograma",
      ayuda: "80 bandas de frecuencia a 22 050 Hz." },
    { id: "perceiver", x: 760, y: 288, w: 200, h: 54, ruta: "audio", nombre: "Condicionamiento + Perceiver",
      ayuda: "Resume el mel en 32 vectores de estilo de 1024 números." },
    { id: "hablante", x: 975, y: 186, w: 176, h: 54, ruta: "audio", nombre: "Codificador de hablante",
      ayuda: "Un vector de 512 números del timbre, a 16 kHz." },
    { id: "gpt", x: 380, y: 330, w: 190, h: 54, ruta: "generacion", nombre: "GPT-2 · 30 capas",
      ayuda: "Lee voz + texto y predice el siguiente código de audio." },
    { id: "tokens", x: 380, y: 506, w: 190, h: 54, ruta: "generacion", nombre: "Tokens de audio",
      ayuda: "Uno por paso; cada uno vuelve a entrar al GPT-2." },
    { id: "dvae", x: 120, y: 506, w: 170, h: 54, forma: "punteada", ruta: "generacion", nombre: "DVAE",
      ayuda: "Rama didáctica: muestra qué describen los tokens. XTTS solo lo usa al entrenar." },
    { id: "hifigan", x: 840, y: 476, w: 200, h: 54, ruta: "salida", nombre: "Vocoder HiFi-GAN",
      ayuda: "Latentes del GPT-2 + vector de hablante → onda." },
    { id: "salida", x: 840, y: 580, w: 190, h: 44, forma: "pastilla", ruta: "salida", nombre: "Audio clonado",
      ayuda: "La voz final a 24 000 Hz." },
  ];

  // Conexiones: puntos de la polilínea, nodo de llegada y etiqueta opcional.
  const ARISTAS = [
    { id: "entrada-texto", a: "texto", puntos: [[380, 91], [380, 121]] },
    { id: "texto-gpt", a: "gpt", puntos: [[380, 175], [380, 303]], etiqueta: ["[≤ 402] fichas de texto", 390, 240, "start"] },
    { id: "audio-mel", a: "mel", puntos: [[860, 94], [860, 122], [760, 122], [760, 159]], etiqueta: ["22 050 Hz", 806, 115, "middle"] },
    { id: "audio-hablante", a: "hablante", puntos: [[860, 94], [860, 122], [975, 122], [975, 159]], etiqueta: ["16 kHz", 920, 115, "middle"] },
    { id: "mel-perceiver", a: "perceiver", puntos: [[760, 213], [760, 261]] },
    { id: "perceiver-gpt", a: "gpt", puntos: [[660, 288], [600, 288], [600, 322], [475, 322]], etiqueta: ["[32, 1024] vectores de estilo", 652, 278, "end"] },
    { id: "gpt-tokens", a: "tokens", puntos: [[350, 357], [350, 479]] },
    { id: "tokens-gpt", a: "gpt", punteada: true, puntos: [[410, 479], [410, 357]], etiqueta: ["se realimenta", 418, 422, "start"] },
    { id: "tokens-dvae", a: "dvae", punteada: true, puntos: [[285, 506], [205, 506]] },
    { id: "gpt-hifigan", a: "hifigan", puntos: [[475, 340], [560, 340], [560, 470], [740, 470]], etiqueta: ["latentes del GPT-2", 570, 460, "start"] },
    { id: "hablante-hifigan", a: "hifigan", puntos: [[975, 213], [975, 482], [940, 482]], etiqueta: ["vector de hablante (512)", 966, 395, "end"] },
    { id: "hifigan-salida", a: "salida", puntos: [[840, 503], [840, 558]] },
  ];

  const ESTADOS = { espera: "en espera", activo: "en curso", listo: "listo", omitido: "omitido", error: "con error" };
  const elementos = {};   // id de nodo -> {grupo, dato, estado}
  const caminos = {};     // id de arista -> path
  const ultimos = {};     // id de nodo -> último evento recibido
  let alPulsar = () => {};
  let ultimoBucle = 0;

  function crear(nombre, atributos, padre) {
    const el = document.createElementNS(SVG, nombre);
    Object.entries(atributos || {}).forEach(([k, v]) => el.setAttribute(k, v));
    if (padre) padre.appendChild(el);
    return el;
  }

  function construir() {
    const svg = crear("svg", { viewBox: "0 0 1100 630", class: "arquitectura__svg", role: "group",
      "aria-label": "Arquitectura de XTTS-v2: bloques y conexiones por donde pasan los datos" });
    const defs = crear("defs", {}, svg);
    const flecha = crear("marker", { id: "flecha", viewBox: "0 0 10 10", refX: "9", refY: "5",
      markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" }, defs);
    crear("path", { d: "M0,0 L10,5 L0,10 z", class: "arista__punta" }, flecha);

    GRUPOS.forEach(function (g) {
      const grupo = crear("g", { class: "grupo", "data-ruta": g.ruta }, svg);
      crear("rect", { x: g.x, y: g.y, width: g.w, height: g.h, rx: 12, class: "grupo__fondo" }, grupo);
      crear("text", { x: g.x + g.w / 2, y: g.y + 18, class: "grupo__titulo", "text-anchor": "middle" }, grupo)
        .textContent = g.titulo;
    });

    ARISTAS.forEach(function (a) {
      const d = a.puntos.map((p, i) => (i ? "L" : "M") + p[0] + "," + p[1]).join(" ");
      const ruta = NODOS.find((n) => n.id === a.a).ruta;
      const camino = crear("path", { d, class: "arista" + (a.punteada ? " arista--punteada" : ""),
        "data-ruta": ruta, "marker-end": "url(#flecha)" }, svg);
      caminos[a.id] = camino;
      if (a.etiqueta) {
        const [texto, x, y, ancla] = a.etiqueta;
        crear("text", { x, y, class: "arista__etiqueta", "text-anchor": ancla }, svg).textContent = texto;
      }
    });

    NODOS.forEach(function (n) {
      const grupo = crear("g", { class: "nodo nodo--espera" + (n.forma ? " nodo--" + n.forma : ""),
        "data-bloque": n.id, "data-ruta": n.ruta, tabindex: "0", role: "button",
        "aria-label": n.nombre + ": " + n.ayuda }, svg);
      crear("rect", { x: n.x - n.w / 2, y: n.y - n.h / 2, width: n.w, height: n.h,
        rx: n.forma === "pastilla" ? n.h / 2 : 8, class: "nodo__caja" }, grupo);
      const conDetalle = n.forma !== "pastilla" || n.id !== "entrada";
      crear("text", { x: n.x, y: n.y + (conDetalle ? -6 : 5), class: "nodo__nombre", "text-anchor": "middle" }, grupo)
        .textContent = n.nombre;
      const dato = crear("text", { x: n.x, y: n.y + 13, class: "nodo__dato", "text-anchor": "middle" }, grupo);
      dato.textContent = n.id === "entrada" ? "" : ESTADOS.espera;
      elementos[n.id] = { grupo, dato, nodo: n };

      const enfocar = () => mostrarDetalle(n.id);
      grupo.addEventListener("mouseenter", enfocar);
      grupo.addEventListener("focus", enfocar);
      grupo.addEventListener("click", () => alPulsar(n.id));
      grupo.addEventListener("keydown", function (evento) {
        if (evento.key === "Enter" || evento.key === " ") {
          evento.preventDefault();
          alPulsar(n.id);
        }
      });
    });
    contenedor.appendChild(svg);
  }

  function textoDato(evento) {
    const partes = [ESTADOS[evento.estado] || evento.estado];
    if (evento.estado === "listo" && evento.ms) {
      partes[0] += " · " + (evento.ms >= 1000 ? (evento.ms / 1000).toFixed(1) + " s" : Math.round(evento.ms) + " ms");
    }
    if (evento.forma && evento.estado === "listo") partes.push("[" + evento.forma.join(", ") + "]");
    return partes.join(" · ");
  }

  function mostrarDetalle(id) {
    const n = elementos[id].nodo;
    const ultimo = ultimos[id];
    let texto = n.nombre + ". " + n.ayuda;
    if (ultimo) {
      texto += " Ahora: " + textoDato(ultimo) + (ultimo.nota ? " · " + ultimo.nota : "") + ".";
    }
    detalle.textContent = texto;
  }

  // Un punto que recorre una conexión. Sin movimiento reducido: solo se ilumina la conexión.
  function recorrer(idArista, duracion) {
    const camino = caminos[idArista];
    if (!camino) return;
    camino.classList.add("arista--viva");
    setTimeout(() => camino.classList.remove("arista--viva"), duracion + 250);
    if (reducido()) return;
    const largo = camino.getTotalLength();
    const punto = crear("circle", { r: 6, class: "punto", "data-ruta": camino.dataset.ruta }, camino.ownerSVGElement);
    const inicio = performance.now();
    function cuadro(ahora) {
      const t = Math.min(1, (ahora - inicio) / duracion);
      const p = camino.getPointAtLength(largo * (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2));
      punto.setAttribute("cx", p.x);
      punto.setAttribute("cy", p.y);
      if (t < 1) requestAnimationFrame(cuadro);
      else punto.remove();
    }
    requestAnimationFrame(cuadro);
  }

  function estado(evento) {
    const el = elementos[evento.id];
    if (!el) return;
    ultimos[evento.id] = evento;
    el.grupo.setAttribute("class", el.grupo.getAttribute("class").replace(/nodo--(espera|activo|listo|omitido|error)/, "nodo--" + evento.estado));
    el.dato.textContent = textoDato(evento);
    if (evento.id === "texto" && evento.estado === "activo") {
      marcar("entrada", "listo");
      recorrer("entrada-texto", 600);
    }
    if (evento.estado === "activo") {
      ARISTAS.filter((a) => a.a === evento.id && !a.punteada).forEach((a) => recorrer(a.id, 700));
      if (evento.id === "dvae") recorrer("tokens-dvae", 700);
    }
  }

  function marcar(id, valor) {
    const el = elementos[id];
    el.grupo.setAttribute("class", el.grupo.getAttribute("class").replace(/nodo--(espera|activo|listo|omitido|error)/, "nodo--" + valor));
  }

  // Cada token generado da una vuelta GPT-2 → tokens → GPT-2 (como mucho una cada 300 ms).
  function paso(numero) {
    elementos.tokens.dato.textContent = (numero + 1) + " tokens";
    const ahora = performance.now();
    if (ahora - ultimoBucle < 300) return;
    ultimoBucle = ahora;
    recorrer("gpt-tokens", 260);
    setTimeout(() => recorrer("tokens-gpt", 260), 260);
  }

  function reiniciar() {
    Object.keys(ultimos).forEach((k) => delete ultimos[k]);
    Object.values(elementos).forEach(function (el) {
      el.grupo.setAttribute("class", el.grupo.getAttribute("class").replace(/nodo--(espera|activo|listo|omitido|error)/, "nodo--espera"));
      el.dato.textContent = el.nodo.id === "entrada" ? "" : ESTADOS.espera;
    });
    detalle.textContent = "Pulsa un bloque para ir a su detalle.";
  }

  // Los bloques que quedaron en curso al detenerse o fallar.
  function interrumpir(valor) {
    Object.entries(elementos).forEach(function ([id, el]) {
      if (!el.grupo.getAttribute("class").includes("nodo--activo")) return;
      marcar(id, valor);
      el.dato.textContent = valor === "error" ? ESTADOS.error : "detenido";
    });
  }

  construir();
  window.Arquitectura = {
    bloques: NODOS.filter((n) => n.id !== "entrada").map((n) => ({ id: n.id, nombre: n.nombre, ruta: n.ruta })),
    estado, paso, reiniciar, interrumpir,
    alPulsar: (fn) => (alPulsar = fn),
    rutaColor: (ruta) => color(RUTAS[ruta]),
  };
})();
