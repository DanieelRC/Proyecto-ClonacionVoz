/* Etapa 3: el GPT-2 genera tokens de audio y la página los recibe en vivo por SSE.
 * Cada paso trae el token, su probabilidad y una fila de atención por capa
 * observada (8 bits + su máximo). Una corrida guardada se reproduce sin servidor.
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

  const autorizado = () => elemento("consentimiento").checked;
  const enCurso = () => !!controlador || temporizador !== null;

  // ------------------------------------------------------------ escala de color

  // Magma aproximado por tramos; la potencia 0.4 revela pesos pequeños (como la etapa 2).
  const PARADAS = [[0, 0, 0, 4], [0.13, 28, 16, 68], [0.25, 79, 18, 123], [0.38, 129, 37, 129],
    [0.5, 181, 54, 122], [0.63, 229, 80, 100], [0.75, 251, 135, 97], [0.88, 254, 194, 135],
    [1, 252, 253, 191]];
  const PALETA = new Uint8ClampedArray(256 * 4);
  for (let q = 0; q < 256; q++) {
    const t = Math.pow(q / 255, 0.4);
    let k = 1;
    while (k < PARADAS.length - 1 && PARADAS[k][0] < t) k++;
    const [t0, ...a] = PARADAS[k - 1];
    const [t1, ...b] = PARADAS[k];
    const f = (t - t0) / (t1 - t0);
    for (let c = 0; c < 3; c++) PALETA[q * 4 + c] = a[c] + (b[c] - a[c]) * f;
    PALETA[q * 4 + 3] = 255;
  }

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
    botonDetener.disabled = !enCurso();
    botonGuardar.disabled = !vista || vista.modo !== "vivo" || !vista.fin || enCurso();
    archivoCorrida.disabled = enCurso();
    elemento("etiqueta-reproducir").classList.toggle("boton--inactivo", enCurso());
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
  }

  function limpiarVista() {
    // Los audios de una corrida reproducida no corresponden a la referencia actual.
    if (vista?.modo === "reproduccion") audiosFase = {};
    else delete audiosFase.tokens_texto;
    detenerTodo();
    vista = null;
    elemento("resultados-generacion").classList.add("oculto");
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
    actualizarProgreso();
    programarDibujo();
  }

  function marcarPestana() {
    Array.from(elemento("pestanas-capas").children).forEach(function (pestana, i) {
      pestana.setAttribute("aria-selected", String(i === vista.capa));
    });
  }

  function agregarPaso(evento) {
    const inicio = vista.inicio;
    const filas = inicio.capas.map((capa) => {
      const fila = evento.atencion[String(capa)];
      return { max: Number(fila.max), q: decodificar(fila.q) };
    });
    vista.pasos.push({ paso: evento.paso, token: evento.token, probabilidad: evento.probabilidad,
      ms: evento.ms, filas });

    filas.forEach(function (fila, i) {
      const n = fila.q.length;
      if (!n) return;
      const imagen = new ImageData(n, 1);
      for (let x = 0; x < n; x++) imagen.data.set(PALETA.subarray(fila.q[x] * 4, fila.q[x] * 4 + 4), x * 4);
      vista.mapas[i].getContext("2d").putImageData(imagen, 0, evento.paso);
    });

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
      (fin.ms_total / 1000).toFixed(1) + " s." + (vista.modo === "vivo" ? " Ya puedes escuchar la fase 4." : "");
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
      if (vista) dibujar();
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

  new MutationObserver(programarDibujo).observe(document.documentElement,
    { attributes: true, attributeFilter: ["data-tema"] });
  window.addEventListener("resize", programarDibujo);

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
    const corrida = { formato: FORMATO, version: 1, guardada: new Date().toISOString(),
      eventos: vista.eventos, audios };
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

  mensaje.textContent = explicarEspera();
  prepararFases();
  actualizarControles();
})();
