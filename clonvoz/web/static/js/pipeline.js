/* Etapas 5 y 6: una corrida completa, en vivo por WebSocket o grabada.
 * En vivo, POST /api/inferencia la lanza y /ws/corrida/<id> entrega cada evento
 * cuando ocurre. Grabada, los mismos eventos salen de un archivo y se reproducen
 * con pausa y velocidad. En ambos casos se anima el diagrama, se reparten los
 * datos a los paneles de cada etapa y la página baja sola a la etapa en curso.
 */
(function () {
  "use strict";
  const elemento = (id) => document.getElementById(id);
  const FORMATO = "clonvoz-corrida-completa";
  const Clonvoz = window.Clonvoz;
  const Arquitectura = window.Arquitectura;
  const campo = elemento("texto-corrida");
  const campoEtapa3 = elemento("texto-generacion");
  const botonEjecutar = elemento("boton-corrida");
  const botonDetener = elemento("detener-corrida");
  const botonGuardar = elemento("guardar-corrida-completa");
  const archivo = elemento("archivo-corrida-completa");
  const lista = elemento("lista-grabaciones");
  const botonReproducir = elemento("reproducir-pausa");
  const botonParar = elemento("detener-reproduccion");
  const velocidad = elemento("velocidad");
  const seguir = elemento("seguir-corrida");
  const mensaje = elemento("estado-corrida");
  const barra = elemento("barra-corrida");

  // A qué panel lleva cada bloque, y a dónde baja la página con cada evento.
  const DESTINOS = { audio: "resultados", mel: "figura-mel", perceiver: "titulo-identidad", locutor: "titulo-locutor",
    texto: "resultados-generacion", gpt: "resultados-generacion", tokens: "tokens-audio",
    dvae: "figura-sintesis-mels", hablante: "titulo-sintesis", hifigan: "titulo-sintesis",
    salida: "titulo-sintesis" };
  const SEGUIMIENTO = { etapa1: "resultados", etapa2: "titulo-identidad", locutor: "titulo-locutor", inicio: "titulo-generacion",
    etapa4: "titulo-sintesis", causal: "titulo-causal" };
  const ESTADOS = { espera: "en espera", activo: "en curso", listo: "listo", omitido: "omitido", error: "con error" };
  const NOMBRES = Object.fromEntries(Arquitectura.bloques.map((b) => [b.id, b.nombre]));

  let modo = "vivo";
  let canal = null;          // WebSocket de la corrida en vivo
  let eventos = [];          // lo recibido o cargado, para guardar y reproducir
  let completa = false;      // la corrida en vivo llegó hasta el final
  let hayWav = false;
  let revision = 0;
  let siguiendo = true;      // la página va detrás de la corrida
  let enPantalla = true;     // el diagrama grande está a la vista
  // Reproducción de una corrida grabada.
  const reproduccion = { eventos: [], indice: 0, temporizador: null, activa: false, pausada: false, tiempo: 0, total: 0 };

  const autorizado = () => elemento("consentimiento").checked;
  const reproduciendo = () => reproduccion.activa;
  const enCurso = () => canal !== null || reproduciendo();
  const reducido = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function actualizarControles() {
    const libre = !enCurso() && !Clonvoz.etapa3.ocupada();
    botonEjecutar.disabled = !libre || !hayWav || !autorizado() || !campo.value.trim();
    botonDetener.disabled = canal === null;
    botonGuardar.disabled = enCurso() || !completa;
    botonReproducir.disabled = !reproduccion.eventos.length || canal !== null;
    botonReproducir.textContent = reproduccion.temporizador !== null ? "Pausa" :
      reproduccion.pausada ? "Reanudar" : "Reproducir";
    botonParar.disabled = !reproduciendo();
    lista.disabled = archivo.disabled = enCurso();
    elemento("etiqueta-reproducir-completa").classList.toggle("boton--inactivo", enCurso());
    elemento("modo-vivo").disabled = elemento("modo-grabado").disabled = enCurso();
    actualizarBarra();
  }

  // ------------------------------------------------------------- modos

  function cambiarModo(nuevo) {
    modo = nuevo;
    elemento("modo-vivo").setAttribute("aria-pressed", String(nuevo === "vivo"));
    elemento("modo-grabado").setAttribute("aria-pressed", String(nuevo === "grabado"));
    elemento("controles-vivo").classList.toggle("oculto", nuevo !== "vivo");
    elemento("controles-grabado").classList.toggle("oculto", nuevo !== "grabado");
    if (nuevo === "grabado") cargarLista();
    mensaje.textContent = nuevo === "vivo" ? "Carga o graba una referencia, escribe un texto y pulsa Ejecutar todo." :
      "Elige una grabación: se reproduce sin el modelo, con los tiempos originales.";
    actualizarControles();
  }
  elemento("modo-vivo").addEventListener("click", () => cambiarModo("vivo"));
  elemento("modo-grabado").addEventListener("click", () => cambiarModo("grabado"));

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

  // ------------------------------------------------- seguir la corrida

  // Desplazamiento propio en vez de scrollIntoView suave: el destino se recalcula en
  // cada cuadro, porque los paneles de arriba crecen mientras llegan los datos.
  let desplazamiento = null;
  function irA(id) {
    const destino = elemento(id);
    if (!destino) return;
    if (desplazamiento) cancelAnimationFrame(desplazamiento);
    const margen = 72;  // la barra fija
    const objetivo = () => Math.max(0, destino.getBoundingClientRect().top + window.scrollY - margen);
    // Con la pestaña oculta no hay cuadros de animación: se salta directo.
    if (reducido() || document.hidden) {
      window.scrollTo(0, objetivo());
      return;
    }
    const desde = window.scrollY;
    const inicio = performance.now();
    const duracion = 650;
    function cuadro(ahora) {
      const t = Math.min(1, (ahora - inicio) / duracion);
      const suave = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      window.scrollTo(0, desde + (objetivo() - desde) * suave);
      desplazamiento = t < 1 ? requestAnimationFrame(cuadro) : null;
    }
    desplazamiento = requestAnimationFrame(cuadro);
  }

  function seguirA(tipo) {
    if (!enCurso() || !seguir.checked || !siguiendo || !SEGUIMIENTO[tipo]) return;
    irA(SEGUIMIENTO[tipo]);
  }

  // Si la persona se mueve por su cuenta, se deja de seguir hasta que lo pida.
  function soltar() {
    if (!enCurso() || !siguiendo) return;
    siguiendo = false;
    cancelarDesplazamiento();
    actualizarBarra();
  }
  window.addEventListener("wheel", soltar, { passive: true });
  window.addEventListener("touchmove", soltar, { passive: true });
  window.addEventListener("keydown", function (evento) {
    if (["PageUp", "PageDown", "ArrowUp", "ArrowDown", "Home", "End", " "].includes(evento.key) &&
        !/^(INPUT|TEXTAREA|SELECT)$/.test(evento.target.tagName)) soltar();
  });
  elemento("volver-a-seguir").addEventListener("click", function () {
    siguiendo = true;
    const ultimo = [...eventos].reverse().find((e) => SEGUIMIENTO[e.tipo]);
    if (ultimo) irA(SEGUIMIENTO[ultimo.tipo]);
    actualizarBarra();
  });
  elemento("ir-al-diagrama").addEventListener("click", () => irA("panel-corrida"));

  // --------------------------------------------- barra fija de avance

  const puntos = {};
  Arquitectura.bloques.forEach(function (b) {
    const li = document.createElement("li");
    const boton = document.createElement("button");
    boton.type = "button";
    boton.className = "barra-corrida__punto";
    boton.dataset.ruta = b.ruta;
    boton.dataset.estado = "espera";
    boton.title = b.nombre;
    boton.setAttribute("aria-label", b.nombre + ": en espera");
    boton.addEventListener("click", () => irA(DESTINOS[b.id]));
    li.appendChild(boton);
    elemento("barra-puntos").appendChild(li);
    puntos[b.id] = boton;
  });

  new IntersectionObserver(function (entradas) {
    enPantalla = entradas[0].isIntersecting;
    actualizarBarra();
  }, { threshold: 0.15 }).observe(elemento("arquitectura"));

  function actualizarBarra() {
    const visible = !enPantalla && (enCurso() || eventos.length > 0);
    barra.classList.toggle("barra-corrida--visible", visible);
    barra.setAttribute("aria-hidden", String(!visible));
    elemento("volver-a-seguir").classList.toggle("oculto", siguiendo || !enCurso() || !seguir.checked);
  }

  function pintarPunto(datos) {
    const boton = puntos[datos.id];
    if (!boton) return;
    boton.dataset.estado = datos.estado;
    boton.setAttribute("aria-label", NOMBRES[datos.id] + ": " + (ESTADOS[datos.estado] || datos.estado));
    if (datos.estado === "activo") elemento("barra-texto").textContent = NOMBRES[datos.id] + " en curso";
  }

  function reiniciarDiagrama() {
    Arquitectura.reiniciar();
    Object.values(puntos).forEach(function (boton) {
      boton.dataset.estado = "espera";
    });
    elemento("barra-texto").textContent = "";
  }

  function interrumpir(estado) {
    Arquitectura.interrumpir(estado);
    Object.values(puntos).forEach(function (boton) {
      if (boton.dataset.estado === "activo") boton.dataset.estado = estado;
    });
  }

  Arquitectura.alPulsar((id) => irA(DESTINOS[id]));

  // ----------------------------------------------------------- eventos

  function aplicar(evento, enVivo) {
    const { tipo, datos, t } = evento;
    if (tipo === "corrida") {
      reiniciarDiagrama();
      Clonvoz.locutor.limpiar();
      siguiendo = true;
      if (!enVivo && datos.texto) {
        campo.value = datos.texto;
        campo.dispatchEvent(new Event("input"));
      }
    } else if (tipo === "bloque") {
      Arquitectura.estado(datos);
      pintarPunto(datos);
      if (datos.id === "tokens" && datos.estado === "activo") mensaje.textContent = "El GPT-2 está generando tokens de audio…";
    } else if (tipo === "etapa1") {
      Clonvoz.etapa1.mostrar(datos, enVivo);
    } else if (tipo === "etapa2") {
      Clonvoz.etapa2.mostrar(datos, enVivo);
    } else if (tipo === "locutor") {
      Clonvoz.locutor.mostrar(datos);
      elemento("barra-texto").textContent = "Locutor: " + datos.locutor + " (" + datos.nombre_modelo + ")";
    } else if (tipo === "inicio") {
      Clonvoz.etapa3.comenzar(enVivo ? "vivo" : "reproduccion");
      Clonvoz.etapa3.aplicar(tipo, datos, t);
    } else if (tipo === "paso") {
      Clonvoz.etapa3.aplicar(tipo, datos, t);
      Arquitectura.paso(datos.paso);
      elemento("barra-texto").textContent = "GPT-2 · " + (datos.paso + 1) + " tokens";
    } else if (tipo === "fin") {
      Clonvoz.etapa3.aplicar(tipo, datos, t);
    } else if (tipo === "etapa4") {
      Clonvoz.etapa3.terminar();
      Clonvoz.etapa4.mostrar(datos, enVivo);
      mensaje.textContent = "Voz clonada lista: " + datos.duracion_s.toFixed(1) + " s.";
      elemento("barra-texto").textContent = "Voz clonada lista";
    } else if (tipo === "explicando") {
      mensaje.textContent = "Voz lista. Midiendo el efecto de bloquear la atención…";
      elemento("barra-texto").textContent = "Midiendo el efecto causal…";
    } else if (tipo === "causal") {
      Clonvoz.etapa3.mostrarCausal(datos);
    } else if (tipo === "terminada") {
      Clonvoz.etapa3.terminar();
      if (datos.motivo !== "completa") interrumpir("omitido");
      mensaje.textContent = datos.motivo === "completa" ? "Corrida completa en " + (t / 1000).toFixed(1) + " s." :
        "Corrida detenida antes de terminar.";
      elemento("barra-texto").textContent = datos.motivo === "completa" ? "Corrida completa" : "Detenida";
      return datos.motivo === "completa";
    } else if (tipo === "error") {
      Clonvoz.etapa3.terminar();
      interrumpir("error");
      mensaje.textContent = (datos.errores || ["No se pudo completar la corrida."]).join(" ");
      elemento("barra-texto").textContent = "Error";
    }
    seguirA(tipo);
    return false;
  }

  // ----------------------------------------------------------- en vivo

  function cerrarCanal() {
    revision += 1;
    if (canal) canal.close();
    canal = null;
    Clonvoz.etapa3.terminar();
  }

  botonEjecutar.addEventListener("click", async function () {
    if (botonEjecutar.disabled) return;
    const referencia = Clonvoz.referencia();
    if (!referencia) return;
    pararReproduccion();
    cerrarCanal();
    const version = revision;
    eventos = [];
    completa = false;
    reiniciarDiagrama();
    Clonvoz.etapa1.invalidar();
    mensaje.textContent = "Enviando el audio y el texto…";
    // Ocupa el canal desde ya, para que los controles reflejen la corrida.
    canal = { close: function () {} };
    siguiendo = true;
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
      mensaje.textContent = "No se pudo contactar al servidor. Prueba el modo Grabado.";
      actualizarControles();
      return;
    }

    mensaje.textContent = "Corrida en marcha. La primera vez se cargan los modelos.";
    const socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + datos.ws);
    canal = socket;
    let terminada = false;
    socket.addEventListener("message", function (mensajeWs) {
      if (version !== revision) return;
      const evento = JSON.parse(mensajeWs.data);
      eventos.push(evento);
      terminada = terminada || evento.tipo === "terminada" || evento.tipo === "error";
      try {
        if (aplicar(evento, true)) completa = true;
      } catch (error) {
        console.error(error);
      }
    });
    socket.addEventListener("close", function () {
      if (version !== revision) return;
      canal = null;
      Clonvoz.etapa3.terminar();
      if (!terminada) {
        interrumpir("error");
        mensaje.textContent = "El canal se cerró antes de terminar. Prueba el modo Grabado.";
      }
      actualizarControles();
    });
  });

  botonDetener.addEventListener("click", function () {
    cancelarDesplazamiento();
    if (canal && canal.readyState === WebSocket.OPEN) {
      // El servidor para en el siguiente token y cierra el canal con su último evento.
      canal.send(JSON.stringify({ accion: "detener" }));
      mensaje.textContent = "Deteniendo… el servidor para al siguiente token.";
    } else {
      cerrarCanal();
      mensaje.textContent = "Corrida detenida.";
    }
    actualizarControles();
  });
  window.addEventListener("pagehide", cerrarCanal);

  botonGuardar.addEventListener("click", function () {
    const corrida = { formato: FORMATO, version: 1, guardada: new Date().toISOString(), eventos };
    const enlace = document.createElement("a");
    enlace.href = URL.createObjectURL(new Blob([JSON.stringify(corrida)], { type: "application/json" }));
    const id = eventos[0]?.datos?.id_corrida || "corrida";
    enlace.download = "corrida-completa-" + String(id).slice(0, 8) + ".json";
    enlace.click();
    setTimeout(() => URL.revokeObjectURL(enlace.href), 1000);
  });

  // ------------------------------------------------------- modo grabado

  async function cargarLista() {
    try {
      const respuesta = await fetch("/api/grabaciones");
      const datos = await respuesta.json();
      const elegida = lista.value;
      lista.replaceChildren(new Option(datos.grabaciones.length ? "Elige una grabación…" :
        "No hay grabaciones en el servidor", ""));
      datos.grabaciones.forEach((g) => lista.add(new Option(g.nombre + " · " + g.kb + " KB", g.nombre)));
      lista.value = elegida;
    } catch (error) {
      // Sin servidor el archivo local sigue funcionando.
      lista.replaceChildren(new Option("Sin servidor: abre un archivo", ""));
    }
  }

  // Duración "reproducible" de cada evento: las esperas largas (cargar el modelo) se recortan.
  const espera = (i) => i === 0 ? 0 :
    Math.min(1000, Math.max(0, (reproduccion.eventos[i].t || 0) - (reproduccion.eventos[i - 1].t || 0)));

  function preparar(corrida, origen) {
    if (corrida.formato !== FORMATO || !Array.isArray(corrida.eventos) || corrida.eventos[0]?.tipo !== "corrida") {
      mensaje.textContent = "Ese archivo no es una corrida completa guardada.";
      return;
    }
    pararReproduccion();
    reproduccion.eventos = corrida.eventos;
    reproduccion.total = corrida.eventos.reduce((suma, _, i) => suma + espera(i), 0);
    mensaje.textContent = "Lista: " + origen + ". Pulsa Reproducir.";
    actualizarControles();
    actualizarProgreso();
  }

  lista.addEventListener("change", async function () {
    if (!lista.value) return;
    mensaje.textContent = "Cargando la grabación…";
    try {
      const respuesta = await fetch("/api/grabaciones/" + encodeURIComponent(lista.value));
      if (!respuesta.ok) throw new Error();
      preparar(await respuesta.json(), lista.value);
    } catch (error) {
      mensaje.textContent = "No se pudo cargar esa grabación.";
    }
  });

  archivo.addEventListener("change", async function () {
    const elegido = archivo.files[0];
    archivo.value = "";
    if (!elegido) return;
    try {
      preparar(JSON.parse(await elegido.text()), elegido.name);
    } catch (error) {
      mensaje.textContent = "Ese archivo no es una corrida completa guardada.";
    }
  });

  function actualizarProgreso() {
    const { tiempo, total } = reproduccion;
    elemento("progreso-reproduccion").style.width = (total ? Math.min(100, (100 * tiempo) / total) : 0) + "%";
    elemento("tiempo-reproduccion").textContent = (tiempo / 1000).toFixed(1) + " / " + (total / 1000).toFixed(1) + " s";
  }

  function siguiente() {
    const r = reproduccion;
    reproduccion.temporizador = null;
    try {
      aplicar(r.eventos[r.indice], false);
    } catch (error) {
      console.error(error);
      mensaje.textContent = "La grabación está dañada en el evento " + r.indice + ".";
      pararReproduccion(false);
      return;
    }
    r.indice += 1;
    if (r.indice >= r.eventos.length) {
      r.pausada = false;
      r.activa = false;
      Clonvoz.etapa3.terminar();
      r.tiempo = r.total;
      actualizarProgreso();
      actualizarControles();
      return;
    }
    programar();
  }

  function programar() {
    const r = reproduccion;
    const demora = espera(r.indice);
    r.tiempo += demora;
    actualizarProgreso();
    r.temporizador = setTimeout(siguiente, demora / Number(velocidad.value));
  }

  function cancelarDesplazamiento() {
    if (desplazamiento) cancelAnimationFrame(desplazamiento);
    desplazamiento = null;
  }

  function pararReproduccion(limpiar = true) {
    cancelarDesplazamiento();
    const r = reproduccion;
    if (r.temporizador !== null) clearTimeout(r.temporizador);
    r.temporizador = null;
    r.activa = false;
    r.pausada = false;
    r.indice = 0;
    r.tiempo = 0;
    Clonvoz.etapa3.terminar();
    if (limpiar) actualizarProgreso();
    actualizarControles();
  }

  botonReproducir.addEventListener("click", function () {
    const r = reproduccion;
    if (r.temporizador !== null) {           // pausa
      clearTimeout(r.temporizador);
      r.temporizador = null;
      r.pausada = true;
      r.tiempo -= espera(r.indice);          // el evento pendiente aún no se mostró
      mensaje.textContent = "En pausa.";
    } else if (r.pausada) {                  // reanudar
      r.pausada = false;
      mensaje.textContent = "Reproduciendo la grabación, sin servidor ni modelo.";
      programar();
    } else {                                 // desde el principio
      cerrarCanal();
      eventos = r.eventos;
      completa = false;
      r.indice = 0;
      r.tiempo = 0;
      r.activa = true;
      siguiendo = true;
      mensaje.textContent = "Reproduciendo la grabación, sin servidor ni modelo.";
      programar();
    }
    actualizarControles();
  });

  botonParar.addEventListener("click", function () {
    pararReproduccion();
    interrumpir("omitido");
    mensaje.textContent = "Reproducción detenida.";
  });

  seguir.addEventListener("change", function () {
    siguiendo = seguir.checked;
    actualizarBarra();
  });

  campo.value = campoEtapa3.value;
  cambiarModo("vivo");
})();
