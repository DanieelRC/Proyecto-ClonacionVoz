/* Etapa 5: una sola corrida de principio a fin, escuchada por WebSocket.
 * POST /api/inferencia la lanza; el canal /ws/corrida/<id> entrega cada evento en
 * el momento en que ocurre. Aquí se ilumina el diagrama de ocho bloques y se
 * reparten los datos a los paneles de cada etapa, que los pintan igual que cuando
 * se ejecutan por separado. Una corrida guardada se reproduce sin servidor.
 */
(function () {
  "use strict";
  const elemento = (id) => document.getElementById(id);
  const FORMATO = "clonvoz-corrida-completa";
  const Clonvoz = window.Clonvoz;
  const campo = elemento("texto-corrida");
  const campoEtapa3 = elemento("texto-generacion");
  const botonEjecutar = elemento("boton-corrida");
  const botonDetener = elemento("detener-corrida");
  const botonGuardar = elemento("guardar-corrida-completa");
  const archivo = elemento("archivo-corrida-completa");
  const mensaje = elemento("estado-corrida");
  const bloques = Object.fromEntries(Array.from(elemento("diagrama").children)
    .map((li) => [li.dataset.bloque, li]));
  // A qué panel lleva cada bloque al pulsarlo.
  const DESTINOS = { audio: "figura-forma_de_onda", mel: "figura-mel", perceiver: "titulo-identidad",
    gpt: "titulo-generacion", tokens: "tokens-audio", dvae: "figura-sintesis-mels",
    hifigan: "titulo-sintesis", salida: "audio-sintesis" };
  const ESTADOS = { activo: "en curso", listo: "listo", omitido: "omitido", error: "con error" };

  let canal = null;        // WebSocket de la corrida en vivo
  let temporizador = null; // reproducción de una corrida guardada
  let eventos = [];        // lo recibido, para poder guardarlo
  let completa = false;    // la corrida mostrada llegó hasta el final
  let hayWav = false;
  let revision = 0;

  const autorizado = () => elemento("consentimiento").checked;
  const enCurso = () => canal !== null || temporizador !== null;

  function actualizarControles() {
    const libre = !enCurso() && !Clonvoz.etapa3.ocupada();
    botonEjecutar.disabled = !libre || !hayWav || !autorizado() || !campo.value.trim();
    botonDetener.disabled = !enCurso();
    botonGuardar.disabled = enCurso() || !completa;
    archivo.disabled = enCurso();
    elemento("etiqueta-reproducir-completa").classList.toggle("boton--inactivo", enCurso());
  }

  // El texto es el mismo en la corrida completa y en la etapa 3 suelta.
  campo.addEventListener("input", function () {
    campoEtapa3.value = campo.value;
    campoEtapa3.dispatchEvent(new Event("input"));
    actualizarControles();
  });
  campoEtapa3.addEventListener("input", function () {
    if (campo.value !== campoEtapa3.value) campo.value = campoEtapa3.value;
    actualizarControles();
  });
  document.addEventListener("referencia-cambiada", function (evento) {
    hayWav = !!evento.detail?.wav;
    actualizarControles();
  });
  elemento("consentimiento").addEventListener("change", actualizarControles);
  ["etapa1-completada", "etapa2-completada", "etapa3-completada", "etapa3-reiniciada"]
    .forEach((nombre) => document.addEventListener(nombre, () => setTimeout(actualizarControles, 0)));

  // ------------------------------------------------------------------ diagrama

  function reiniciarDiagrama() {
    Object.values(bloques).forEach(function (li) {
      li.className = "bloque";
      li.querySelector(".bloque__dato").textContent = "";
      li.querySelector(".bloque__estado").textContent = "en espera";
    });
  }

  function pintarBloque(datos) {
    const li = bloques[datos.id];
    if (!li) return;
    li.className = "bloque bloque--" + datos.estado;
    let estado = ESTADOS[datos.estado] || datos.estado;
    if (datos.estado === "listo" && datos.ms) {
      estado += " · " + (datos.ms >= 1000 ? (datos.ms / 1000).toFixed(1) + " s" : Math.round(datos.ms) + " ms");
    }
    li.querySelector(".bloque__estado").textContent = estado;
    const partes = [];
    if (datos.forma) partes.push("[" + datos.forma.join(", ") + "]");
    if (datos.nota) partes.push(datos.nota);
    if (partes.length || datos.estado !== "activo") li.querySelector(".bloque__dato").textContent = partes.join(" · ");
  }

  function marcarInterrumpidos(estado) {
    Object.values(bloques).forEach(function (li) {
      if (!li.classList.contains("bloque--activo")) return;
      li.className = "bloque bloque--" + estado;
      li.querySelector(".bloque__estado").textContent = estado === "error" ? "con error" : "detenido";
    });
  }

  Object.entries(bloques).forEach(function ([id, li]) {
    li.querySelector("button").addEventListener("click", function () {
      const destino = elemento(DESTINOS[id]);
      const reducido = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (destino) destino.scrollIntoView({ behavior: reducido ? "auto" : "smooth", block: "start" });
    });
  });

  // ------------------------------------------------------------------- eventos

  function aplicar(evento, enVivo) {
    const { tipo, datos, t } = evento;
    if (tipo === "corrida") {
      reiniciarDiagrama();
      if (!enVivo && datos.texto) {
        campo.value = datos.texto;
        campo.dispatchEvent(new Event("input"));
      }
    } else if (tipo === "bloque") {
      pintarBloque(datos);
      if (datos.id === "tokens" && datos.estado === "activo") mensaje.textContent = "El GPT-2 está generando tokens de audio…";
    } else if (tipo === "etapa1") {
      Clonvoz.etapa1.mostrar(datos, enVivo);
    } else if (tipo === "etapa2") {
      Clonvoz.etapa2.mostrar(datos, enVivo);
    } else if (tipo === "inicio") {
      Clonvoz.etapa3.comenzar(enVivo ? "vivo" : "reproduccion");
      Clonvoz.etapa3.aplicar(tipo, datos, t);
    } else if (tipo === "paso") {
      Clonvoz.etapa3.aplicar(tipo, datos, t);
      const li = bloques.tokens;
      li.querySelector(".bloque__dato").textContent = (datos.paso + 1) + " tokens";
    } else if (tipo === "fin") {
      Clonvoz.etapa3.aplicar(tipo, datos, t);
    } else if (tipo === "etapa4") {
      Clonvoz.etapa3.terminar();
      Clonvoz.etapa4.mostrar(datos, enVivo);
      mensaje.textContent = "Voz clonada lista: " + datos.duracion_s.toFixed(1) + " s. Ya puedes escucharla abajo.";
    } else if (tipo === "explicando") {
      mensaje.textContent = "Voz clonada lista. Midiendo ahora el efecto de bloquear la atención…";
    } else if (tipo === "causal") {
      Clonvoz.etapa3.mostrarCausal(datos);
    } else if (tipo === "terminada") {
      Clonvoz.etapa3.terminar();
      completa = datos.motivo === "completa";
      if (!completa) marcarInterrumpidos("omitido");
      mensaje.textContent = completa ? "Corrida completa en " + (t / 1000).toFixed(1) +
        " s. Cada panel de abajo tiene su detalle." : "Corrida detenida antes de terminar.";
    } else if (tipo === "error") {
      Clonvoz.etapa3.terminar();
      marcarInterrumpidos("error");
      mensaje.textContent = (datos.errores || ["No se pudo completar la corrida."]).join(" ");
    }
  }

  function cerrar() {
    revision += 1;
    if (canal) canal.close();
    canal = null;
    if (temporizador !== null) clearTimeout(temporizador);
    temporizador = null;
    Clonvoz.etapa3.terminar();
  }

  // ----------------------------------------------------------- corrida en vivo

  botonEjecutar.addEventListener("click", async function () {
    if (botonEjecutar.disabled) return;
    const referencia = Clonvoz.referencia();
    if (!referencia) return;
    cerrar();
    const version = revision;
    eventos = [];
    completa = false;
    reiniciarDiagrama();
    Clonvoz.etapa1.invalidar();
    mensaje.textContent = "Enviando el audio y el texto…";
    // Ocupa el canal desde ya, para que los controles reflejen la corrida.
    canal = { close: function () {} };
    actualizarControles();

    const formulario = new FormData();
    formulario.append("consentimiento", "si");
    formulario.append("audio", referencia.wav, referencia.nombre);
    formulario.append("texto", campo.value);
    formulario.append("determinista", elemento("determinista").checked ? "si" : "no");
    formulario.append("semilla", String(Math.floor(Math.random() * 2 ** 31)));
    let datos;
    try {
      const respuesta = await fetch("/api/inferencia", { method: "POST", body: formulario });
      datos = await respuesta.json().catch(() => ({}));
      if (version !== revision) return;
      if (!respuesta.ok) {
        canal = null;
        mensaje.textContent = (datos.errores || ["No se pudo iniciar la corrida."]).join(" ");
        actualizarControles();
        return;
      }
    } catch (error) {
      if (version !== revision) return;
      canal = null;
      mensaje.textContent = "No se pudo contactar al servidor. Revisa la terminal.";
      actualizarControles();
      return;
    }

    mensaje.textContent = "Corrida en marcha. La primera vez se cargan los modelos.";
    const esquema = location.protocol === "https:" ? "wss://" : "ws://";
    const socket = new WebSocket(esquema + location.host + datos.ws);
    canal = socket;
    let terminada = false;
    socket.addEventListener("message", function (mensajeWs) {
      if (version !== revision) return;
      const evento = JSON.parse(mensajeWs.data);
      eventos.push(evento);
      terminada = terminada || evento.tipo === "terminada" || evento.tipo === "error";
      try {
        aplicar(evento, true);
      } catch (error) {
        console.error(error);
      }
    });
    socket.addEventListener("close", function () {
      if (version !== revision) return;
      canal = null;
      Clonvoz.etapa3.terminar();
      if (!terminada) {
        marcarInterrumpidos("error");
        mensaje.textContent = "El canal se cerró antes de terminar. Revisa la terminal.";
      }
      actualizarControles();
    });
  });

  botonDetener.addEventListener("click", function () {
    if (temporizador !== null) {
      cerrar();
      marcarInterrumpidos("omitido");
      mensaje.textContent = "Reproducción detenida.";
    } else if (canal && canal.readyState === WebSocket.OPEN) {
      // El servidor para en el siguiente token y cierra el canal con su último evento.
      canal.send(JSON.stringify({ accion: "detener" }));
      mensaje.textContent = "Deteniendo… el servidor para al siguiente token.";
    } else {
      cerrar();
      mensaje.textContent = "Corrida detenida.";
    }
    actualizarControles();
  });
  window.addEventListener("pagehide", cerrar);

  // ------------------------------------------------- guardar y reproducir corrida

  botonGuardar.addEventListener("click", function () {
    const corrida = { formato: FORMATO, version: 1, guardada: new Date().toISOString(), eventos };
    const enlace = document.createElement("a");
    enlace.href = URL.createObjectURL(new Blob([JSON.stringify(corrida)], { type: "application/json" }));
    const id = eventos[0]?.datos?.id_corrida || "corrida";
    enlace.download = "corrida-completa-" + String(id).slice(0, 8) + ".json";
    enlace.click();
    setTimeout(() => URL.revokeObjectURL(enlace.href), 1000);
  });

  archivo.addEventListener("change", async function () {
    const elegido = archivo.files[0];
    archivo.value = "";
    if (!elegido) return;
    let corrida;
    try {
      corrida = JSON.parse(await elegido.text());
      if (corrida.formato !== FORMATO || !Array.isArray(corrida.eventos) ||
          corrida.eventos[0]?.tipo !== "corrida") throw new Error("formato");
    } catch (error) {
      mensaje.textContent = "Ese archivo no es una corrida completa guardada.";
      return;
    }
    cerrar();
    const version = revision;
    eventos = corrida.eventos;
    completa = false;
    reiniciarDiagrama();
    mensaje.textContent = "Reproduciendo una corrida guardada, sin servidor ni modelo.";
    let i = 0;
    function siguiente() {
      if (version !== revision) return;
      try {
        aplicar(eventos[i], false);
      } catch (error) {
        temporizador = null;
        Clonvoz.etapa3.terminar();
        mensaje.textContent = "La corrida guardada está dañada en el evento " + i + ".";
        actualizarControles();
        return;
      }
      i += 1;
      if (i >= eventos.length) {
        temporizador = null;
        Clonvoz.etapa3.terminar();
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

  campo.value = campoEtapa3.value;
  mensaje.textContent = "Carga o graba una referencia, escribe un texto y pulsa Ejecutar todo.";
  actualizarControles();
})();
