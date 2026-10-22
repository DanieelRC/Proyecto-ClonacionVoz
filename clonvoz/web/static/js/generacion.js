/* Etapa 3: el GPT-2 genera tokens de audio y la página los recibe en vivo por SSE.
 * Cada paso trae el token, su probabilidad, una fila de atención por capa
 * observada (8 bits + su máximo) y el reparto de cada cabeza de cada capa entre
 * voz, texto y audio. Después se puede pedir el mapa completo de cualquier cabeza
 * y medir qué pasa al bloquear la atención. Una corrida guardada se reproduce sin
 * servidor.
 */
(function () {
  "use strict";
  const elemento = (id) => document.getElementById(id);
  const FORMATO = "clonvoz-corrida-etapa3";
  const LIMITE = 239;
  const campo = elemento("texto-generacion");
  const botonGenerar = elemento("boton-generar");
  const botonDetener = elemento("boton-detener");
  const botonGuardar = elemento("guardar-corrida");
  const archivoCorrida = elemento("archivo-corrida");
  const mensaje = elemento("estado-generacion");
  const lienzo = elemento("mapa-generacion");
  const lienzoZoom = elemento("zoom-texto");

  let referencia = null;       // datos de la etapa 1
  let identidadLista = false;  // la etapa 2 terminó para esa referencia
  let wavReferencia = null;
  let urlOriginal = null;
  let controlador = null;      // generación en vivo
  let temporizador = null;     // reproducción de una corrida
  let revision = 0;
  let vista = null;            // lo que se está mostrando, en vivo o reproducido
  let audiosFase = {};         // fase -> data URL del WAV
  let dibujoPendiente = false;
  let externa = false;         // la corrida completa (etapa 5) está alimentando esta vista

  const autorizado = () => elemento("consentimiento").checked;
  const enCurso = () => !!controlador || temporizador !== null || externa;

  // ------------------------------------------------------------ escala de color

  // Magma aproximado por tramos; la potencia 0.4 revela pesos pequeños (como la etapa 2).
  const PARADAS = [[0, 0, 0, 4], [0.13, 28, 16, 68], [0.25, 79, 18, 123], [0.38, 129, 37, 129],
    [0.5, 181, 54, 122], [0.63, 229, 80, 100], [0.75, 251, 135, 97], [0.88, 254, 194, 135],
    [1, 252, 253, 191]];
  function construirPaleta(posicion) {
    const paleta = new Uint8ClampedArray(256 * 4);
    for (let q = 0; q < 256; q++) {
      const t = posicion(Math.pow(q / 255, 0.4));
      let k = 1;
      while (k < PARADAS.length - 1 && PARADAS[k][0] < t) k++;
      const [t0, ...a] = PARADAS[k - 1];
      const [t1, ...b] = PARADAS[k];
      const f = (t - t0) / (t1 - t0);
      for (let c = 0; c < 3; c++) paleta[q * 4 + c] = a[c] + (b[c] - a[c]) * f;
      paleta[q * 4 + 3] = 255;
    }
    return paleta;
  }
  // En claro la escala va invertida y sin el crema final (0.88), igual que MAPAS_CALOR en graficas.py.
  const PALETAS = {
    claro: construirPaleta((t) => 0.88 * (1 - t)),
    oscuro: construirPaleta((t) => t),
  };
  const paletaActual = () =>
    PALETAS[document.documentElement.getAttribute("data-tema")] || PALETAS.claro;

  const color = (nombre) => getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();

  function decodificar(base64) {
    const binario = atob(base64);
    const bytes = new Uint8Array(binario.length);
    for (let i = 0; i < binario.length; i++) bytes[i] = binario.charCodeAt(i);
    return bytes;
  }

  // ------------------------------------------------------------------- controles

  function actualizarControles() {
    const texto = campo.value.trim();
    elemento("contador-texto").textContent = texto.length + " / " + LIMITE;
    botonGenerar.disabled = !referencia || !identidadLista || !autorizado() || !texto || enCurso();
    botonDetener.disabled = !enCurso() || externa;  // la corrida completa se detiene desde su panel
    botonGuardar.disabled = !vista || vista.modo !== "vivo" || !vista.fin || enCurso();
    archivoCorrida.disabled = enCurso();
    elemento("etiqueta-reproducir").classList.toggle("boton--inactivo", enCurso());
    const servidor = hayServidor();
    const conSeleccion = servidor && !!vista.seleccion;
    botonMapaCabeza.disabled = !conSeleccion || !!botonMapaCabeza.dataset.cargando;
    botonCabezasCausal.disabled = !conSeleccion || !!botonCabezasCausal.dataset.cargando;
    botonCausal.disabled = !servidor || !!botonCausal.dataset.cargando;
    actualizarFases();
  }

  function explicarEspera() {
    if (!referencia) return "Completa la etapa 1 y la etapa 2 para habilitar la etapa 3.";
    if (!identidadLista) return "Ejecuta la etapa 2 para esta referencia: el GPT-2 usa sus 32 vectores.";
    return "Identidad lista. Escribe un texto y pulsa Generar.";
  }

  function detenerTodo() {
    revision += 1;
    if (controlador) controlador.abort();
    controlador = null;
    if (temporizador !== null) clearTimeout(temporizador);
    temporizador = null;
    externa = false;
  }

  function limpiarVista() {
    // Los audios de una corrida reproducida no corresponden a la referencia actual.
    if (vista?.modo === "reproduccion") audiosFase = {};
    else delete audiosFase.tokens_texto;
    detenerTodo();
    vista = null;
    elemento("resultados-generacion").classList.add("oculto");
    limpiarExplicaciones();
    document.dispatchEvent(new CustomEvent("etapa3-reiniciada"));
    prepararFases();
  }

  function cambiarReferencia(nueva) {
    if (vista?.modo === "reproduccion") {
      referencia = nueva;
      identidadLista = false;
      actualizarControles();
      return;
    }
    limpiarVista();
    referencia = nueva;
    identidadLista = false;
    audiosFase = {};
    prepararFases();
    mensaje.textContent = explicarEspera();
    actualizarControles();
  }

  document.addEventListener("referencia-cambiada", function (evento) {
    wavReferencia = evento.detail?.wav || null;
    if (urlOriginal) URL.revokeObjectURL(urlOriginal);
    urlOriginal = wavReferencia ? URL.createObjectURL(wavReferencia) : null;
    cambiarReferencia(null);
  });
  document.addEventListener("etapa1-iniciada", () => cambiarReferencia(null));
  document.addEventListener("etapa1-completada", (evento) => cambiarReferencia(evento.detail));
  document.addEventListener("etapa2-completada", function (evento) {
    if (referencia && evento.detail?.id_corrida === referencia.id_corrida) {
      identidadLista = true;
      if (!enCurso()) mensaje.textContent = explicarEspera();
      actualizarControles();
    }
  });
  elemento("consentimiento").addEventListener("change", function () {
    if (!autorizado()) cambiarReferencia(null);
    actualizarControles();
  });
  campo.addEventListener("input", actualizarControles);

  // ------------------------------------------------------------------- la vista

  function iniciarVista(inicio, modo) {
    const ancho = inicio.prefijo + inicio.tope;
    vista = {
      modo, inicio, eventos: [], pasos: [], fin: null,
      zonas: null, seleccion: null, extras: { mapas: {}, cabezas: null, causal: null },
      capa: inicio.capas.length - 1,
      mapas: inicio.capas.map(function () {
        const c = document.createElement("canvas");
        c.width = ancho;
        c.height = inicio.tope;
        return c;
      }),
    };
    elemento("resultados-generacion").classList.remove("oculto");
    elemento("resumen-generacion").replaceChildren();
    elemento("lectura-atencion").textContent = "";

    const fichas = elemento("fichas-bpe");
    fichas.replaceChildren();
    inicio.piezas.forEach(function (pieza, i) {
      const ficha = document.createElement("span");
      ficha.className = "ficha";
      ficha.textContent = pieza;
      ficha.title = "id " + inicio.ids[i];
      fichas.appendChild(ficha);
    });

    const pestanas = elemento("pestanas-capas");
    pestanas.replaceChildren();
    inicio.capas.forEach(function (capa, i) {
      const pestana = document.createElement("button");
      pestana.type = "button";
      pestana.className = "pestana";
      pestana.setAttribute("role", "tab");
      pestana.textContent = "Capa " + capa + " de " + inicio.total_capas;
      pestana.addEventListener("click", function () {
        vista.capa = i;
        marcarPestana();
        programarDibujo();
      });
      pestanas.appendChild(pestana);
    });
    marcarPestana();
    elemento("tokens-audio").replaceChildren();
    limpiarExplicaciones();
    if (inicio.cabezas) vista.seleccion = { capa: inicio.total_capas - 1, cabeza: null };
    actualizarProgreso();
    programarDibujo();
  }

  function marcarPestana() {
    Array.from(elemento("pestanas-capas").children).forEach(function (pestana, i) {
      pestana.setAttribute("aria-selected", String(i === vista.capa));
    });
  }

  // Relativa: cada fila a su propio máximo. Absoluta: q / 255 × máximo es el peso real,
  // entre 0 y 1, así que el mismo color significa el mismo peso en cualquier paso.
  function pintarFila(i, paso, fila, paleta) {
    const n = fila.q.length;
    if (!n) return;
    const absoluta = elemento("escala-absoluta").checked;
    const imagen = new ImageData(n, 1);
    for (let x = 0; x < n; x++) {
      const q = absoluta ? Math.min(255, Math.round(fila.q[x] * fila.max)) : fila.q[x];
      imagen.data.set(paleta.subarray(q * 4, q * 4 + 4), x * 4);
    }
    vista.mapas[i].getContext("2d").putImageData(imagen, 0, paso);
  }

  // La paleta depende del tema: al cambiarlo se repintan las filas ya recibidas.
  function repintarMapas() {
    elemento("nota-escala").textContent = elemento("escala-absoluta").checked ?
      "Escala absoluta, de 0 a 1, con potencia 0.4: sirve para comparar pasos entre sí. " +
      "Las filas con la atención muy repartida se ven más apagadas porque sus pesos son de verdad más pequeños." :
      "Cada fila se escala a su propio máximo y el color usa una potencia 0.4 para revelar pesos pequeños: " +
      "sirve para comparar posiciones dentro de un paso, no entre pasos. Los porcentajes sí usan los pesos reales.";
    if (!vista?.pasos) return;
    const paleta = paletaActual();
    vista.pasos.forEach((paso) => paso.filas.forEach((fila, i) => pintarFila(i, paso.paso, fila, paleta)));
    const s = vista.seleccion;
    if (s && vista.extras.mapas[s.capa + ":" + s.cabeza]) dibujarMapaCabeza(vista.extras.mapas[s.capa + ":" + s.cabeza]);
    programarDibujo();
  }

  function agregarPaso(evento) {
    const inicio = vista.inicio;
    const filas = inicio.capas.map((capa) => {
      const fila = evento.atencion[String(capa)];
      return { max: Number(fila.max), q: decodificar(fila.q) };
    });
    vista.pasos.push({ paso: evento.paso, token: evento.token, probabilidad: evento.probabilidad,
      ms: evento.ms, filas });

    const paleta = paletaActual();
    filas.forEach((fila, i) => pintarFila(i, evento.paso, fila, paleta));
    if (evento.zonas && inicio.cabezas) acumularZonas(evento.zonas);

    const caja = elemento("tokens-audio");
    caja.querySelector(".token--reciente")?.classList.remove("token--reciente");
    const token = document.createElement("span");
    token.className = "token token--reciente" + (evento.token >= 1024 ? " token--especial" : "");
    token.textContent = evento.token;
    token.title = "Paso " + evento.paso + (evento.probabilidad === null ? "" :
      " · probabilidad " + evento.probabilidad.toFixed(3));
    caja.appendChild(token);
    caja.scrollTop = caja.scrollHeight;
    actualizarProgreso();
    programarDibujo();
  }

  function actualizarProgreso() {
    const n = vista.pasos.length;
    const tope = vista.inicio.tope;
    elemento("progreso-relleno").style.width = Math.min(100, (100 * n) / tope) + "%";
    const ultimo = vista.pasos[n - 1];
    let texto = "Paso " + n + " de " + tope + " como máximo";
    if (ultimo) texto += " · " + Math.round(ultimo.ms) + " ms el último";
    if (ultimo && ultimo.probabilidad !== null) texto += " · probabilidad " + ultimo.probabilidad.toFixed(3);
    elemento("progreso-texto").textContent = texto;
  }

  const MOTIVOS = {
    parada: "El modelo emitió su token de fin.",
    tope: "Llegó al tope de tokens sin emitir el fin.",
    cancelada: "Detenida antes de terminar.",
  };

  function mostrarFin(fin) {
    vista.fin = fin;
    const inicio = vista.inicio;
    const filas = [
      ["Tokens generados", fin.total_tokens],
      ["Fin", MOTIVOS[fin.motivo] || fin.motivo],
      ["Por token", fin.ms_por_token + " ms"],
      ["Total", (fin.ms_total / 1000).toFixed(1) + " s"],
      ["Secuencia inicial", inicio.prefijo + " posiciones (32 voz + " + (inicio.prefijo - 33) + " texto + 1)"],
      ["Elección", inicio.determinista ? "la más probable" :
        "muestreo, temperatura " + inicio.muestreo.temperature + ", semilla " + inicio.semilla],
    ];
    const resumen = elemento("resumen-generacion");
    resumen.replaceChildren();
    filas.forEach(function ([nombre, valor]) {
      const grupo = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = nombre;
      dd.textContent = valor;
      grupo.append(dt, dd);
      resumen.appendChild(grupo);
    });
    mensaje.textContent = (MOTIVOS[fin.motivo] || "") + " " + fin.total_tokens + " tokens en " +
      (fin.ms_total / 1000).toFixed(1) + " s." +
      (vista.modo === "vivo" ? " Ya puedes escuchar la fase 4 y sintetizar la voz en la etapa 4." : "");
    explicarSeleccion();
    if (vista.modo === "vivo" && fin.total_tokens > 0 && fin.motivo !== "cancelada")
      document.dispatchEvent(new CustomEvent("etapa3-completada",
        { detail: { id_corrida: vista.inicio.id_corrida, texto: vista.inicio.texto } }));
  }

  function aplicar(tipo, datos) {
    if (tipo === "inicio") iniciarVista(datos, vista?.modo || "vivo");
    else if (tipo === "paso" && vista) agregarPaso(datos);
    else if (tipo === "fin" && vista) mostrarFin(datos);
    else if (tipo === "error") mensaje.textContent = (datos.errores || ["No se pudo completar la etapa 3."]).join(" ");
  }

  // --------------------------------------------------------------------- dibujo

  function programarDibujo() {
    if (dibujoPendiente) return;
    dibujoPendiente = true;
    requestAnimationFrame(function () {
      dibujoPendiente = false;
      if (vista?.inicio) dibujar();
    });
  }

  function ajustar(canvas, ancho, alto, fijarAncho) {
    const escala = window.devicePixelRatio || 1;
    if (fijarAncho) canvas.style.width = ancho + "px";
    canvas.style.height = alto + "px";
    canvas.width = Math.round(ancho * escala);
    canvas.height = Math.round(alto * escala);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(escala, 0, 0, escala, 0, 0);
    ctx.imageSmoothingEnabled = false;
    return ctx;
  }

  function dibujar() {
    const { inicio, pasos, capa } = vista;
    const n = pasos.length;
    const columnas = inicio.prefijo + Math.max(n - 1, 0);
    const [t0, t1] = inicio.zonas.texto;
    const fondo = color("--superficie-hundida");

    // Secuencia completa: una banda de zonas arriba y el mapa debajo.
    const ancho = lienzo.clientWidth;
    const alto = 260;
    const banda = 8;
    let ctx = ajustar(lienzo, ancho, alto, false);
    ctx.fillStyle = fondo;
    ctx.fillRect(0, 0, ancho, alto);
    const x = (col) => (col / columnas) * ancho;
    [[0, 32, "--ruta-audio"], [t0, t1, "--ruta-texto"], [t1, columnas, "--ruta-generacion"]]
      .forEach(function ([a, b, nombre]) {
        ctx.fillStyle = color(nombre);
        ctx.fillRect(x(a), 0, Math.max(x(b) - x(a), 1), banda);
      });
    if (n) ctx.drawImage(vista.mapas[capa], 0, 0, columnas, n, 0, banda + 2, ancho, alto - banda - 2);

    // Solo el texto, con una columna ancha por ficha para poder leerla.
    const etiquetas = ["inicio", ...inicio.piezas, "fin"];
    const celda = Math.max(26, Math.floor(lienzoZoom.parentElement.clientWidth / etiquetas.length));
    const anchoZoom = celda * etiquetas.length;
    const cabeza = 58;
    const altoZoom = 240;
    ctx = ajustar(lienzoZoom, anchoZoom, altoZoom, true);
    ctx.fillStyle = fondo;
    ctx.fillRect(0, 0, anchoZoom, altoZoom);
    if (n) ctx.drawImage(vista.mapas[capa], t0, 0, t1 - t0, n, 0, cabeza, anchoZoom, altoZoom - cabeza);

    const ultima = n ? pasos[n - 1].filas[capa] : null;
    let maximo = -1;
    if (ultima) {
      for (let c = t0; c < t1; c++) if (maximo < 0 || ultima.q[c] > ultima.q[maximo]) maximo = c;
    }
    ctx.font = "12px " + color("--fuente");
    etiquetas.forEach(function (etiqueta, i) {
      ctx.save();
      ctx.translate(i * celda + celda / 2 + 4, cabeza - 6);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = t0 + i === maximo ? color("--texto") : color("--texto-suave");
      ctx.font = (t0 + i === maximo ? "600 " : "") + "12px " + color("--fuente");
      ctx.fillText(etiqueta, 0, 0);
      ctx.restore();
    });
    if (maximo >= 0) {
      ctx.strokeStyle = color("--texto");
      ctx.lineWidth = 2;
      ctx.strokeRect((maximo - t0) * celda + 1, cabeza, celda - 2, altoZoom - cabeza);
    }
    if (ultima) explicarPaso(ultima, t1, etiquetas[maximo - t0]);
    dibujarRejilla();
    if (vista.zonas && vista.seleccion && !vista.fin) explicarSeleccion();
    dibujarCausal();
  }

  function explicarPaso(fila, t1, pieza) {
    // Los porcentajes usan el peso real: q / 255 × máximo de la fila.
    const suma = [0, 0, 0];
    for (let c = 0; c < fila.q.length; c++) {
      suma[c < 32 ? 0 : c < t1 ? 1 : 2] += (fila.q[c] / 255) * fila.max;
    }
    const total = suma[0] + suma[1] + suma[2] || 1;
    const pct = (v) => ((100 * v) / total).toLocaleString("es-MX", { maximumFractionDigits: 1 }) + " %";
    const capa = vista.inicio.capas[vista.capa];
    elemento("lectura-atencion").textContent = "Último paso (" + (vista.pasos.length - 1) +
      "), capa " + capa + ": " + pct(suma[0]) + " a la voz, " + pct(suma[1]) + " al texto y " +
      pct(suma[2]) + " a los tokens de audio ya generados. Dentro del texto, la ficha más atendida es «" +
      pieza + "».";
  }

  new MutationObserver(repintarMapas).observe(document.documentElement,
    { attributes: true, attributeFilter: ["data-tema"] });
  window.addEventListener("resize", programarDibujo);


  // ------------------------------------------- todas las capas y todas las cabezas

  const lienzoRejilla = elemento("rejilla-cabezas");
  const botonMapaCabeza = elemento("boton-mapa-cabeza");
  const botonCabezasCausal = elemento("boton-cabezas-causal");
  const botonCausal = elemento("boton-causal");
  const ZONAS = [["voz", "--ruta-audio", "a la voz"], ["texto", "--ruta-texto", "al texto"],
    ["audio", "--ruta-generacion", "a los tokens de audio"]];
  const CELDA = 24;        // ancho de una cabeza
  const FILA = 14;         // alto de una capa
  const MARGEN_REJILLA = 30;

  function rgb(hex) {
    const n = parseInt(hex.replace("#", ""), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  // Suma de un paso a la media por cabeza: zonas es [capas × cabezas × 3] en 8 bits, escala absoluta.
  function acumularZonas(base64) {
    const { total_capas: capas, cabezas } = vista.inicio;
    const q = decodificar(base64);
    if (q.length !== capas * cabezas * 3) return;
    if (!vista.zonas) vista.zonas = { suma: new Float64Array(q.length), pasos: 0 };
    for (let i = 0; i < q.length; i++) vista.zonas.suma[i] += q[i] / 255;
    vista.zonas.pasos += 1;
  }

  // Reparto medio [voz, texto, audio] de una cabeza, o de toda la capa si cabeza es null.
  function repartoDe(capa, cabeza) {
    const { cabezas } = vista.inicio;
    const { suma, pasos } = vista.zonas;
    const salida = [0, 0, 0];
    const desde = cabeza === null ? 0 : cabeza;
    const hasta = cabeza === null ? cabezas : cabeza + 1;
    for (let h = desde; h < hasta; h++)
      for (let z = 0; z < 3; z++) salida[z] += suma[(capa * cabezas + h) * 3 + z];
    return salida.map((v) => v / (pasos * (hasta - desde)));
  }

  function dibujarRejilla() {
    if (!vista?.inicio?.cabezas) return;
    const { total_capas: capas, cabezas } = vista.inicio;
    const ancho = MARGEN_REJILLA + (cabezas + 1) * CELDA + 8;
    const alto = 18 + capas * FILA;
    const ctx = ajustar(lienzoRejilla, ancho, alto, true);
    ctx.fillStyle = color("--superficie");
    ctx.fillRect(0, 0, ancho, alto);
    ctx.font = "10px " + color("--fuente");
    ctx.fillStyle = color("--texto-tenue");
    ctx.textAlign = "center";
    for (let h = 0; h < cabezas; h++) ctx.fillText(String(h + 1), MARGEN_REJILLA + h * CELDA + CELDA / 2, 11);
    ctx.fillText("μ", MARGEN_REJILLA + cabezas * CELDA + 4 + CELDA / 2, 11);
    ctx.textAlign = "right";
    for (let c = 0; c < capas; c++)
      if (c === 0 || (c + 1) % 5 === 0) ctx.fillText(String(c + 1), MARGEN_REJILLA - 5, 18 + c * FILA + 11);
    if (!vista.zonas) return;
    const colores = ZONAS.map(([, variable]) => rgb(color(variable)));
    for (let c = 0; c < capas; c++) {
      for (let h = 0; h <= cabezas; h++) {
        const reparto = repartoDe(c, h === cabezas ? null : h);
        const mezcla = [0, 1, 2].map((k) => Math.round(reparto.reduce((s, v, z) => s + v * colores[z][k], 0)));
        ctx.fillStyle = "rgb(" + mezcla.join(",") + ")";
        ctx.fillRect(MARGEN_REJILLA + h * CELDA + (h === cabezas ? 4 : 0), 18 + c * FILA, CELDA - 1, FILA - 1);
      }
    }
    const s = vista.seleccion;
    if (s) {
      const h = s.cabeza === null ? cabezas : s.cabeza;
      ctx.strokeStyle = color("--texto");
      ctx.lineWidth = 2;
      ctx.strokeRect(MARGEN_REJILLA + h * CELDA + (h === cabezas ? 4 : 0) - 1, 18 + s.capa * FILA - 1, CELDA + 1, FILA + 1);
    }
    dibujarDona();
  }

  // Gráfica circular de la celda elegida: el mismo reparto que la frase de abajo, en porcentajes.
  function dibujarDona() {
    const lienzoDona = elemento("dona-cabeza");
    if (!lienzoDona) return;  // plantilla anterior aún en caché: reinicia el servidor
    const visible = !!(vista?.zonas && vista.seleccion);
    lienzoDona.parentElement.classList.toggle("oculto", !visible);
    if (!visible) return;
    const reparto = repartoDe(vista.seleccion.capa, vista.seleccion.cabeza);
    const total = reparto[0] + reparto[1] + reparto[2] || 1;
    const ancho = 320;
    const alto = 340;
    const ctx = ajustar(lienzoDona, ancho, alto, true);
    ctx.fillStyle = color("--superficie");
    ctx.fillRect(0, 0, ancho, alto);
    const cx = ancho / 2;
    const cy = 122;
    const radio = 104;
    let angulo = -Math.PI / 2;
    reparto.forEach(function (valor, z) {
      const fin = angulo + (valor / total) * 2 * Math.PI;
      if (valor > 0) {
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, radio, angulo, fin);
        ctx.closePath();
        ctx.fillStyle = color(ZONAS[z][1]);
        ctx.fill();
        // Un filo del color del fondo separa las porciones sin depender solo del tono.
        ctx.strokeStyle = color("--superficie");
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      angulo = fin;
    });
    ctx.beginPath();
    ctx.arc(cx, cy, radio * 0.58, 0, 2 * Math.PI);
    ctx.fillStyle = color("--superficie");
    ctx.fill();

    const s = vista.seleccion;
    ctx.textAlign = "center";
    ctx.fillStyle = color("--texto");
    ctx.font = "600 15px " + color("--fuente");
    ctx.fillText("Capa " + (s.capa + 1), cx, cy - 2);
    ctx.fillStyle = color("--texto-suave");
    ctx.font = "13px " + color("--fuente");
    ctx.fillText(s.cabeza === null ? "promedio (μ)" : "cabeza " + (s.cabeza + 1), cx, cy + 17);

    const nombres = ["voz", "texto", "tokens de audio"];
    ctx.textAlign = "left";
    reparto.forEach(function (valor, z) {
      const y = 256 + z * 26;
      ctx.fillStyle = color(ZONAS[z][1]);
      ctx.fillRect(48, y - 11, 13, 13);
      ctx.fillStyle = color("--texto");
      ctx.font = "600 14px " + color("--fuente");
      ctx.fillText(porciento(valor / total), 70, y);
      ctx.fillStyle = color("--texto-suave");
      ctx.font = "14px " + color("--fuente");
      ctx.fillText(nombres[z], 128, y);
    });
    lienzoDona.setAttribute("aria-label", "Reparto de la atención de la " + nombreSeleccion(s) + ": " +
      reparto.map((v, z) => porciento(v / total) + " " + ZONAS[z][2]).join(", ") + ".");
  }

  const porciento = (v) => (100 * v).toLocaleString("es-MX", { maximumFractionDigits: 1 }) + " %";
  const nombreSeleccion = (s) => "capa " + (s.capa + 1) +
    (s.cabeza === null ? " (promedio de sus cabezas)" : ", cabeza " + (s.cabeza + 1));

  function explicarSeleccion() {
    const salida = elemento("lectura-cabeza");
    if (!vista?.zonas || !vista.seleccion) {
      salida.textContent = vista?.zonas ? "" :
        "Esta corrida no trae el reparto por cabeza (se guardó con una versión anterior).";
      return;
    }
    const reparto = repartoDe(vista.seleccion.capa, vista.seleccion.cabeza);
    const orden = [0, 1, 2].sort((a, b) => reparto[b] - reparto[a]);
    salida.textContent = "En la " + nombreSeleccion(vista.seleccion) + ", al elegir cada token: " +
      orden.map((z) => porciento(reparto[z]) + " de su atención va " + ZONAS[z][2]).join(", ") +
      ". Media de " + vista.zonas.pasos + " pasos, en escala absoluta.";
  }

  function seleccionar(capa, cabeza) {
    vista.seleccion = { capa, cabeza };
    explicarSeleccion();
    dibujarRejilla();
    const clave = capa + ":" + cabeza;
    if (vista.extras.mapas[clave]) dibujarMapaCabeza(vista.extras.mapas[clave]);
    actualizarControles();
  }

  lienzoRejilla.addEventListener("click", function (evento) {
    if (!vista?.zonas) return;
    const { total_capas: capas, cabezas } = vista.inicio;
    const caja = lienzoRejilla.getBoundingClientRect();
    const capa = Math.floor((evento.clientY - caja.top - 18) / FILA);
    let cabeza = Math.floor((evento.clientX - caja.left - MARGEN_REJILLA) / CELDA);
    if (capa < 0 || capa >= capas || cabeza < 0 || cabeza > cabezas) return;
    seleccionar(capa, cabeza === cabezas ? null : cabeza);
  });
  lienzoRejilla.addEventListener("keydown", function (evento) {
    if (!vista?.zonas) return;
    const { total_capas: capas, cabezas } = vista.inicio;
    const s = vista.seleccion || { capa: capas - 1, cabeza: 0 };
    let capa = s.capa;
    let columna = s.cabeza === null ? cabezas : s.cabeza;
    if (evento.key === "ArrowUp") capa -= 1;
    else if (evento.key === "ArrowDown") capa += 1;
    else if (evento.key === "ArrowLeft") columna -= 1;
    else if (evento.key === "ArrowRight") columna += 1;
    else return;
    evento.preventDefault();
    capa = Math.max(0, Math.min(capas - 1, capa));
    columna = Math.max(0, Math.min(cabezas, columna));
    seleccionar(capa, columna === cabezas ? null : columna);
  });

  // El servidor conserva los tokens: se le puede pedir cualquier capa y cabeza.
  const hayServidor = () => vista?.modo === "vivo" && !!vista.fin && vista.fin.total_tokens > 0 &&
    !!referencia && autorizado() && !enCurso();

  async function pedir(ruta, cuerpo) {
    const respuesta = await fetch(ruta, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ consentimiento: "si", id_corrida: referencia.id_corrida, ...cuerpo }),
    });
    const datos = await respuesta.json().catch(() => ({}));
    if (!respuesta.ok) throw new Error((datos.errores || ["No se pudo calcular."]).join(" "));
    return datos;
  }

  function dibujarMapaCabeza(mapa) {
    const lienzoMapa = elemento("mapa-cabeza");
    const inicio = vista.inicio;
    const q = decodificar(mapa.q);
    const imagen = new ImageData(mapa.columnas, mapa.filas);
    const paleta = paletaActual();
    for (let i = 0; i < q.length; i++) imagen.data.set(paleta.subarray(q[i] * 4, q[i] * 4 + 4), i * 4);
    const origen = document.createElement("canvas");
    origen.width = mapa.columnas;
    origen.height = mapa.filas;
    origen.getContext("2d").putImageData(imagen, 0, 0);

    lienzoMapa.classList.remove("oculto");
    const ancho = lienzoMapa.clientWidth;
    const alto = 260;
    const banda = 8;
    const ctx = ajustar(lienzoMapa, ancho, alto, false);
    ctx.fillStyle = color("--superficie-hundida");
    ctx.fillRect(0, 0, ancho, alto);
    const x = (col) => (col / mapa.columnas) * ancho;
    const [t0, t1] = inicio.zonas.texto;
    [[0, 32, "--ruta-audio"], [t0, t1, "--ruta-texto"], [t1, mapa.columnas, "--ruta-generacion"]]
      .forEach(function ([a, b, nombre]) {
        ctx.fillStyle = color(nombre);
        ctx.fillRect(x(a), 0, Math.max(x(b) - x(a), 1), banda);
      });
    ctx.drawImage(origen, 0, 0, mapa.columnas, mapa.filas, 0, banda + 2, ancho, alto - banda - 2);
    elemento("pie-mapa-cabeza").textContent = "Mapa completo de la " +
      nombreSeleccion({ capa: mapa.capa - 1, cabeza: mapa.cabeza === null ? null : mapa.cabeza - 1 }) +
      ": una fila por paso, toda la matriz con una sola escala (su máximo es " +
      mapa.max.toLocaleString("es-MX", { maximumFractionDigits: 3 }) +
      "). El mismo color es el mismo peso en cualquier paso.";
  }

  botonMapaCabeza.addEventListener("click", async function () {
    if (botonMapaCabeza.disabled) return;
    const { capa, cabeza } = vista.seleccion;
    const version = revision;
    const estado = elemento("estado-cabeza");
    estado.textContent = "Repitiendo el cálculo sobre la secuencia generada…";
    botonMapaCabeza.dataset.cargando = "1";
    actualizarControles();
    try {
      const mapa = await pedir("/api/explicar/atencion", { capa: capa + 1, cabeza: cabeza === null ? null : cabeza + 1 });
      if (version !== revision) return;
      vista.extras.mapas[capa + ":" + cabeza] = mapa;
      estado.textContent = "";
      if (vista.seleccion.capa === capa && vista.seleccion.cabeza === cabeza) dibujarMapaCabeza(mapa);
    } catch (error) {
      if (version === revision) estado.textContent = error.message;
    } finally {
      delete botonMapaCabeza.dataset.cargando;
      actualizarControles();
    }
  });

  function mostrarCabezasCausal(datos) {
    const caja = elemento("barras-cabezas");
    caja.replaceChildren();
    const titulo = document.createElement("p");
    titulo.className = "ayuda";
    titulo.textContent = "Capa " + datos.capa + ": cuánto cambia la distribución de salida (divergencia KL media, " +
      "en nats) al apagar cada cabeza por separado. Una cabeza que mira mucho puede importar poco, y al revés.";
    const barras = document.createElement("div");
    barras.className = "barras";
    const maximo = Math.max(...datos.cabezas.map((c) => c.kl_media), 1e-9);
    datos.cabezas.forEach(function (c) {
      const fila = document.createElement("div");
      fila.className = "barras__fila";
      const nombre = document.createElement("span");
      nombre.textContent = "Cabeza " + c.cabeza;
      const pista = document.createElement("span");
      pista.className = "barras__pista";
      const relleno = document.createElement("span");
      relleno.className = "barras__relleno";
      relleno.style.width = Math.max(0, (100 * c.kl_media) / maximo) + "%";
      pista.appendChild(relleno);
      const valor = document.createElement("span");
      valor.textContent = "KL " + c.kl_media.toLocaleString("es-MX", { maximumFractionDigits: 4 });
      fila.append(nombre, pista, valor);
      barras.appendChild(fila);
    });
    caja.append(titulo, barras);
  }

  botonCabezasCausal.addEventListener("click", async function () {
    if (botonCabezasCausal.disabled) return;
    const capa = vista.seleccion.capa;
    const version = revision;
    const estado = elemento("estado-cabeza");
    estado.textContent = "Apagando una a una las cabezas de la capa " + (capa + 1) +
      "… son tantas pasadas del modelo como cabezas; en CPU puede tardar cerca de un minuto.";
    botonCabezasCausal.dataset.cargando = "1";
    actualizarControles();
    try {
      const datos = await pedir("/api/explicar/causal", { capa: capa + 1 });
      if (version !== revision) return;
      vista.extras.cabezas = datos;
      estado.textContent = "";
      mostrarCabezasCausal(datos);
    } catch (error) {
      if (version === revision) estado.textContent = error.message;
    } finally {
      delete botonCabezasCausal.dataset.cargando;
      actualizarControles();
    }
  });

  // -------------------------------------------------------- mirar no es causar

  function dibujarCausal() {
    const datos = vista?.extras?.causal;
    if (!datos) return;
    const lienzoCausal = elemento("grafica-causal");
    const ancho = lienzoCausal.clientWidth;
    const alto = 330;
    const ctx = ajustar(lienzoCausal, ancho, alto, false);
    ctx.fillStyle = color("--superficie");
    ctx.fillRect(0, 0, ancho, alto);
    const izquierda = 46;
    const derecha = 10;
    const n = datos.pasos;
    const x = (i) => izquierda + (n > 1 ? (i / (n - 1)) * (ancho - izquierda - derecha) : 0);
    const todas = Object.fromEntries(datos.intervenciones.filter((i) => i.capas === "todas").map((i) => [i.zona, i]));

    function panel(arriba, altura, titulo, series, minimo, maximo, formato) {
      ctx.font = "600 12px " + color("--fuente");
      ctx.fillStyle = color("--texto");
      ctx.textAlign = "left";
      ctx.fillText(titulo, izquierda, arriba - 8);
      const y = (v) => arriba + altura - ((v - minimo) / (maximo - minimo || 1)) * altura;
      ctx.font = "11px " + color("--fuente");
      ctx.textAlign = "right";
      [minimo, (minimo + maximo) / 2, maximo].forEach(function (marca) {
        ctx.strokeStyle = color("--borde");
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(izquierda, y(marca));
        ctx.lineTo(ancho - derecha, y(marca));
        ctx.stroke();
        ctx.fillStyle = color("--texto-tenue");
        ctx.fillText(formato(marca), izquierda - 6, y(marca) + 4);
      });
      series.forEach(function ([valores, variable]) {
        ctx.strokeStyle = color(variable);
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        valores.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
        ctx.stroke();
      });
    }

    panel(26, 110, "Atención: fracción del peso que va a cada zona (media de capas y cabezas)",
      ZONAS.map(([zona, variable]) => [datos.atencion[zona], variable]), 0, 1, (v) => Math.round(100 * v) + " %");
    const caidas = [todas.voz.caida, todas.texto.caida];
    const maximo = Math.max(1e-6, ...caidas.flat());
    const minimo = Math.min(0, ...caidas.flat());
    panel(190, 110, "Intervención: caída del log p del token elegido al bloquear esa zona (nats)",
      [[todas.voz.caida, "--ruta-audio"], [todas.texto.caida, "--ruta-texto"]], minimo, maximo,
      (v) => v.toLocaleString("es-MX", { maximumFractionDigits: 2 }));
    ctx.font = "11px " + color("--fuente");
    ctx.fillStyle = color("--texto-tenue");
    ctx.textAlign = "center";
    ctx.fillText("paso de generación (0 a " + (n - 1) + ")", (izquierda + ancho - derecha) / 2, alto - 6);
  }

  function mostrarCausal(datos) {
    if (!vista?.inicio) return;
    vista.extras.causal = datos;
    elemento("resultados-causal").classList.remove("oculto");
    elemento("estado-causal").textContent = "";
    const cuerpo = elemento("tabla-causal");
    cuerpo.replaceChildren();
    const veces = (caida) => Math.exp(-caida);
    datos.intervenciones.forEach(function (i) {
      const fila = cuerpo.insertRow();
      if (i.capas === "todas") fila.className = "fila--principal";
      const efecto = veces(i.caida_media);
      [i.zona === "voz" ? "los 32 vectores de voz" : "el texto",
        i.capas === "todas" ? "todas (1 a " + datos.capas + ")" : i.capas + " (" + i.rango[0] + " a " + i.rango[1] + ")",
        porciento(i.atencion_recibida ?? datos.atencion_media[i.zona]),
        (efecto <= 1 ? "se multiplica por " : "sube a ") + efecto.toLocaleString("es-MX", { maximumFractionDigits: 2 }) +
          " (" + (i.caida_media >= 0 ? "−" : "+") + Math.abs(i.caida_media).toLocaleString("es-MX", { maximumFractionDigits: 3 }) + " nats)",
        "en " + porciento(i.cambia_el_mas_probable) + " de los pasos",
      ].forEach((texto) => (fila.insertCell().textContent = texto));
    });
    const todas = Object.fromEntries(datos.intervenciones.filter((i) => i.capas === "todas").map((i) => [i.zona, i]));
    const masMirada = datos.atencion_media.voz >= datos.atencion_media.texto ? "voz" : "texto";
    const masNecesaria = todas.voz.caida_media >= todas.texto.caida_media ? "voz" : "texto";
    const hacia = { voz: "a la voz", texto: "al texto" };
    const de = { voz: "de la voz", texto: "del texto" };
    elemento("lectura-causal").textContent = "En esta corrida el modelo mira más " + hacia[masMirada] +
      " (" + porciento(datos.atencion_media[masMirada]) + " de su atención) y depende más " + de[masNecesaria] +
      ": al bloquear esa zona, la probabilidad media del token que había elegido se multiplica por " +
      veces(todas[masNecesaria].caida_media).toLocaleString("es-MX", { maximumFractionDigits: 2 }) + ". " +
      (masMirada === masNecesaria ? "Aquí atención y dependencia apuntan al mismo lado, pero no en la misma proporción: " :
        "Atención y dependencia no coinciden: ") +
      "la voz recibe " + porciento(datos.atencion_media.voz) + " de la atención y bloquearla cambia el token más probable en " +
      porciento(todas.voz.cambia_el_mas_probable) + " de los pasos; el texto recibe " + porciento(datos.atencion_media.texto) +
      " y cambia " + porciento(todas.texto.cambia_el_mas_probable) + ". La probabilidad es la del modelo antes de los filtros de muestreo.";
    dibujarCausal();
    actualizarControles();
  }

  botonCausal.addEventListener("click", async function () {
    if (botonCausal.disabled) return;
    const version = revision;
    elemento("estado-causal").textContent = "Repitiendo el cálculo con la atención bloqueada… son nueve pasadas " +
      "del modelo sobre toda la secuencia; en CPU tarda decenas de segundos.";
    botonCausal.dataset.cargando = "1";
    actualizarControles();
    try {
      const datos = await pedir("/api/explicar/causal", {});
      if (version === revision) mostrarCausal(datos);
    } catch (error) {
      if (version === revision) elemento("estado-causal").textContent = error.message;
    } finally {
      delete botonCausal.dataset.cargando;
      actualizarControles();
    }
  });

  function limpiarExplicaciones() {
    elemento("lectura-cabeza").textContent = "";
    elemento("estado-cabeza").textContent = "";
    elemento("pie-mapa-cabeza").textContent = "";
    elemento("mapa-cabeza").classList.add("oculto");
    elemento("barras-cabezas").replaceChildren();
    elemento("resultados-causal").classList.add("oculto");
    elemento("estado-causal").textContent = "";
    elemento("tabla-causal").replaceChildren();
    elemento("lectura-causal").textContent = "";
  }

  function restaurarExtras(extras) {
    if (!extras || !vista?.inicio) return;
    if (extras.mapas && typeof extras.mapas === "object") vista.extras.mapas = extras.mapas;
    if (extras.cabezas) {
      vista.extras.cabezas = extras.cabezas;
      mostrarCabezasCausal(extras.cabezas);
    }
    if (extras.causal) mostrarCausal(extras.causal);
    if (vista.seleccion) seleccionar(vista.seleccion.capa, vista.seleccion.cabeza);
  }

  elemento("escala-absoluta").addEventListener("change", repintarMapas);

  // --------------------------------------------------------- generación en vivo

  function leerEvento(bloque) {
    let tipo = "message";
    const datos = [];
    bloque.split("\n").forEach(function (linea) {
      if (linea.startsWith("event:")) tipo = linea.slice(6).trim();
      else if (linea.startsWith("data:")) datos.push(linea.slice(5).trimStart());
    });
    return [tipo, JSON.parse(datos.join("\n"))];
  }

  botonGenerar.addEventListener("click", async function () {
    if (botonGenerar.disabled) return;
    limpiarVista();
    const version = revision;
    controlador = new AbortController();
    vista = { modo: "vivo" };
    actualizarControles();
    mensaje.textContent = "Tokenizando el texto y preparando el GPT-2… la primera vez carga el modelo.";
    let terminado = false;
    try {
      const respuesta = await fetch("/api/generar", {
        method: "POST", headers: { "Content-Type": "application/json" },
        signal: controlador.signal,
        body: JSON.stringify({ consentimiento: "si", id_corrida: referencia.id_corrida,
          texto: campo.value, determinista: elemento("determinista").checked,
          semilla: Math.floor(Math.random() * 2 ** 31) }),
      });
      if (!respuesta.ok) {
        const datos = await respuesta.json().catch(() => ({}));
        if (version !== revision) return;
        if (respuesta.status === 410) cambiarReferencia(null);
        vista = null;
        mensaje.textContent = (datos.errores || ["No se pudo iniciar la etapa 3."]).join(" ");
        return;
      }
      const lector = respuesta.body.getReader();
      const decodificador = new TextDecoder();
      const reloj = performance.now();
      let resto = "";
      mensaje.textContent = "Generando tokens de audio…";
      for (;;) {
        const { value, done } = await lector.read();
        if (done || version !== revision) break;
        resto += decodificador.decode(value, { stream: true });
        let corte;
        while ((corte = resto.indexOf("\n\n")) >= 0) {
          const [tipo, datos] = leerEvento(resto.slice(0, corte));
          resto = resto.slice(corte + 2);
          if (tipo === "inicio") iniciarVista(datos, "vivo");
          else aplicar(tipo, datos);
          vista.eventos?.push({ tipo, t: Math.round(performance.now() - reloj), datos });
          terminado = terminado || tipo === "fin" || tipo === "error";
        }
      }
      if (!terminado && version === revision)
        mensaje.textContent = "La conexión se cerró antes de terminar. Revisa la terminal.";
    } catch (error) {
      if (version !== revision) return;
      mensaje.textContent = error.name === "AbortError" ?
        "Detenida en el paso " + (vista?.pasos?.length || 0) + ". El servidor para al siguiente token." :
        "No se pudo contactar al servidor. Revisa la terminal.";
    } finally {
      if (version === revision) {
        controlador = null;
        actualizarControles();
      }
    }
  });

  botonDetener.addEventListener("click", function () {
    const reproduciendo = temporizador !== null;
    const pasos = vista?.pasos?.length || 0;
    detenerTodo();
    mensaje.textContent = reproduciendo ? "Reproducción detenida." :
      "Detenida en el paso " + pasos + ". El servidor para al siguiente token.";
    actualizarControles();
  });
  window.addEventListener("pagehide", detenerTodo);

  // ------------------------------------------------- guardar y reproducir corrida

  botonGuardar.addEventListener("click", function () {
    const audios = {};
    ["mel", "tokens_referencia", "tokens_texto"].forEach(function (fase) {
      if (audiosFase[fase]) audios[fase] = audiosFase[fase];
    });
    const corrida = { formato: FORMATO, version: 2, guardada: new Date().toISOString(),
      eventos: vista.eventos, audios, extras: vista.extras };
    const enlace = document.createElement("a");
    enlace.href = URL.createObjectURL(new Blob([JSON.stringify(corrida)], { type: "application/json" }));
    enlace.download = "corrida-etapa3-" + vista.inicio.id_corrida.slice(0, 8) + ".json";
    enlace.click();
    setTimeout(() => URL.revokeObjectURL(enlace.href), 1000);
  });

  archivoCorrida.addEventListener("change", async function () {
    const archivo = archivoCorrida.files[0];
    archivoCorrida.value = "";
    if (!archivo) return;
    let corrida;
    try {
      corrida = JSON.parse(await archivo.text());
      if (corrida.formato !== FORMATO || !Array.isArray(corrida.eventos) ||
          corrida.eventos[0]?.tipo !== "inicio") throw new Error("formato");
    } catch (error) {
      mensaje.textContent = "Ese archivo no es una corrida guardada de la etapa 3.";
      return;
    }
    limpiarVista();
    const version = revision;
    vista = { modo: "reproduccion" };
    audiosFase = {};
    Object.entries(corrida.audios || {}).forEach(function ([fase, url]) {
      if (typeof url === "string" && url.startsWith("data:audio/wav;base64,")) audiosFase[fase] = url;
    });
    prepararFases();
    mensaje.textContent = "Reproduciendo una corrida guardada, sin servidor ni modelo.";
    const eventos = corrida.eventos;
    let i = 0;
    function siguiente() {
      if (version !== revision) return;
      try {
        aplicar(eventos[i].tipo, eventos[i].datos);
      } catch (error) {
        temporizador = null;
        mensaje.textContent = "La corrida guardada está dañada en el evento " + i + ".";
        actualizarControles();
        return;
      }
      i += 1;
      if (i >= eventos.length) {
        temporizador = null;
        restaurarExtras(corrida.extras);
        actualizarControles();
        return;
      }
      // Con los tiempos originales, pero sin esperas largas (p. ej. cargar el modelo).
      const espera = Math.min(1000, Math.max(0, (eventos[i].t || 0) - (eventos[i - 1].t || 0)));
      temporizador = setTimeout(siguiente, espera);
    }
    temporizador = setTimeout(siguiente, 0);
    actualizarControles();
  });

  // ---------------------------------------------------------- escuchar por fases

  const fases = Array.from(elemento("lista-fases").querySelectorAll("li[data-fase]"));

  function prepararFases() {
    fases.forEach(function (li) {
      const audio = li.querySelector("audio");
      if (!audio) return;
      const fase = li.dataset.fase;
      audio.pause();
      audio.removeAttribute("src");
      audio.classList.add("oculto");
      const reproduccion = vista?.modo === "reproduccion";
      li.querySelector(".estado-fase").textContent = reproduccion && fase === "original" ?
        "No viene en la corrida: el archivo no guarda tu voz original." : "";
      if (fase === "original" && urlOriginal && !reproduccion) mostrarAudio(li, urlOriginal);
    });
  }

  function mostrarAudio(li, url) {
    const audio = li.querySelector("audio");
    audio.src = url;
    audio.classList.remove("oculto");
  }

  function actualizarFases() {
    const reproduccion = vista?.modo === "reproduccion";
    const vivo = !reproduccion && referencia && autorizado();
    fases.forEach(function (li) {
      const boton = li.querySelector("button");
      if (!boton || boton.dataset.cargando) return;
      const fase = li.dataset.fase;
      let listo;
      if (reproduccion) listo = !!audiosFase[fase];
      else if (fase === "tokens_texto") listo = vivo && !!vista?.fin && vista.fin.total_tokens > 0;
      else if (fase === "tokens_referencia") listo = vivo && !!wavReferencia;
      else listo = vivo;
      boton.disabled = !listo;
      const estado = li.querySelector(".estado-fase");
      if (reproduccion && !listo) estado.textContent = "No viene en esta corrida.";
      else if (estado.textContent === "No viene en esta corrida.") estado.textContent = "";
    });
  }

  fases.forEach(function (li) {
    const boton = li.querySelector("button");
    if (!boton) return;
    const fase = li.dataset.fase;
    const estado = li.querySelector(".estado-fase");
    boton.addEventListener("click", async function () {
      if (audiosFase[fase]) {
        mostrarAudio(li, audiosFase[fase]);
        li.querySelector("audio").play().catch(() => {});
        return;
      }
      if (!referencia) return;
      const version = revision;
      const id = referencia.id_corrida;
      const formulario = new FormData();
      formulario.append("consentimiento", "si");
      formulario.append("fase", fase);
      formulario.append("id_corrida", id);
      if (fase === "tokens_referencia") formulario.append("audio", wavReferencia, "referencia.wav");
      boton.dataset.cargando = "1";
      boton.disabled = true;
      estado.textContent = fase === "mel" ? "Invirtiendo el mel…" : "Decodificando tokens…";
      try {
        const respuesta = await fetch("/api/fases/audio", { method: "POST", body: formulario });
        if (!respuesta.ok) {
          const datos = await respuesta.json().catch(() => ({}));
          estado.textContent = (datos.errores || ["No se pudo generar este audio."]).join(" ");
          return;
        }
        const wav = await respuesta.blob();
        const url = await new Promise(function (resolver, rechazar) {
          const lector = new FileReader();
          lector.onload = () => resolver(lector.result);
          lector.onerror = rechazar;
          lector.readAsDataURL(new Blob([wav], { type: "audio/wav" }));
        });
        if (referencia?.id_corrida !== id || vista?.modo === "reproduccion") return;
        if (fase === "tokens_texto" && version !== revision) return;
        audiosFase[fase] = url;
        estado.textContent = "";
        mostrarAudio(li, url);
      } catch (error) {
        estado.textContent = "No se pudo contactar al servidor.";
      } finally {
        delete boton.dataset.cargando;
        actualizarControles();
      }
    });
  });

  // La corrida completa (etapa 5) alimenta esta vista con los mismos eventos que el SSE.
  window.Clonvoz = window.Clonvoz || {};
  window.Clonvoz.etapa3 = {
    comenzar: function (modo) {
      limpiarVista();
      externa = true;
      vista = { modo };
      if (modo === "reproduccion") {
        audiosFase = {};
        prepararFases();
      }
      mensaje.textContent = modo === "vivo" ? "Generando tokens de audio dentro de la corrida completa…" :
        "Reproduciendo la etapa 3 de una corrida guardada.";
      actualizarControles();
    },
    aplicar: function (tipo, datos, t) {
      if (!externa) return;
      if (tipo === "inicio") iniciarVista(datos, vista?.modo || "vivo");
      else aplicar(tipo, datos);
      vista?.eventos?.push({ tipo, t, datos });
    },
    mostrarCausal: function (datos) {
      if (vista?.inicio) mostrarCausal(datos);
    },
    terminar: function () {
      externa = false;
      actualizarControles();
    },
    ocupada: enCurso,
  };

  repintarMapas();
  mensaje.textContent = explicarEspera();
  prepararFases();
  actualizarControles();
})();
