/* Etapa 4: los tokens de la etapa 3 se convierten en la voz clonada.
 * El servidor ya tiene la identidad, el texto y los tokens; solo le falta el WAV
 * de referencia para el vector de hablante, que no guardó. Se reenvía una vez.
 */
(function () {
  "use strict";
  const elemento = (id) => document.getElementById(id);
  const boton = elemento("boton-sintetizar");
  const botonContraste = elemento("boton-contraste");
  const mensaje = elemento("estado-sintesis");

  let referencia = null;    // datos de la etapa 1
  let wavReferencia = null;
  let tokensListos = false; // la etapa 3 terminó para esa referencia
  let resultado = null;     // lo que se está mostrando
  let enVivo = false;       // el resultado mostrado existe en el servidor
  let ocupado = false;
  let revision = 0;

  const autorizado = () => elemento("consentimiento").checked;

  function actualizarControles() {
    boton.disabled = !referencia || !tokensListos || !wavReferencia || !autorizado() || ocupado;
    botonContraste.disabled = !resultado || !enVivo || !referencia || !autorizado() ||
      !!botonContraste.dataset.cargando;
  }

  function explicarEspera() {
    if (!referencia) return "Completa las etapas 1 a 3 para habilitar la etapa 4.";
    if (!tokensListos) return "Genera los tokens en la etapa 3: la etapa 4 los convierte en voz.";
    return "Tokens listos. Pulsa Sintetizar la voz.";
  }

  function limpiar() {
    revision += 1;
    resultado = null;
    enVivo = false;
    ocupado = false;
    const audio = elemento("audio-sintesis");
    audio.pause();
    audio.removeAttribute("src");
    const contraste = elemento("audio-contraste");
    contraste.pause();
    contraste.removeAttribute("src");
    contraste.classList.add("oculto");
    elemento("estado-contraste").textContent = "";
    elemento("resultados-sintesis").classList.add("oculto");
  }

  function cambiarReferencia(nueva) {
    limpiar();
    referencia = nueva;
    tokensListos = false;
    mensaje.textContent = explicarEspera();
    actualizarControles();
  }

  document.addEventListener("referencia-cambiada", function (evento) {
    wavReferencia = evento.detail?.wav || null;
    cambiarReferencia(null);
  });
  document.addEventListener("etapa1-iniciada", () => cambiarReferencia(null));
  document.addEventListener("etapa1-completada", (evento) => cambiarReferencia(evento.detail));
  document.addEventListener("etapa3-reiniciada", function () {
    // Tokens nuevos en camino: la voz mostrada ya no les corresponde.
    if (!enVivo && resultado) return;  // una corrida reproducida no depende de la etapa 3 en vivo
    limpiar();
    tokensListos = false;
    mensaje.textContent = explicarEspera();
    actualizarControles();
  });
  document.addEventListener("etapa3-completada", function (evento) {
    if (!referencia || evento.detail?.id_corrida !== referencia.id_corrida) return;
    tokensListos = true;
    if (!ocupado && !resultado) mensaje.textContent = explicarEspera();
    actualizarControles();
  });
  elemento("consentimiento").addEventListener("change", function () {
    if (!autorizado()) cambiarReferencia(null);
    actualizarControles();
  });

  function pintarFigura(id, graficas, descripcion) {
    const lienzo = elemento(id).querySelector(".figura__lienzo");
    lienzo.textContent = "";
    ["claro", "oscuro"].forEach(function (tema) {
      const imagen = document.createElement("img");
      imagen.className = "figura__imagen figura__imagen--" + tema;
      imagen.src = graficas[tema];
      imagen.alt = tema === "claro" ? descripcion : "";
      if (tema === "oscuro") imagen.setAttribute("aria-hidden", "true");
      lienzo.appendChild(imagen);
    });
  }

  function mostrar(datos, vivo) {
    resultado = datos;
    enVivo = !!vivo;
    elemento("resultados-sintesis").classList.remove("oculto");
    elemento("texto-sintesis").textContent = datos.texto ? "«" + datos.texto + "»" : "";
    elemento("audio-sintesis").src = datos.audio;
    pintarFigura("figura-sintesis-mels", datos.graficas.mels,
      "Tres mel-espectrogramas con la misma escala: la referencia, el que describen los tokens y el de la voz generada.");
    pintarFigura("figura-sintesis-onda", datos.graficas.onda, "Forma de onda de la voz generada: amplitud contra tiempo.");
    const formas = datos.formas;
    elemento("forma-sintesis-mels").textContent = "mel de salida [" + formas.mel_salida.join(", ") + "]" +
      (formas.mel_dvae ? " · mel del DVAE [" + formas.mel_dvae.join(", ") + "]" : " · sin DVAE (falta dvae.pth)");
    elemento("forma-sintesis-onda").textContent = "tensor [" + datos.muestras + "]";

    const tiempos = datos.tiempos_ms;
    const filas = [
      ["Duración", datos.duracion_s.toFixed(2) + " s a " + datos.sr + " Hz"],
      ["Latentes del GPT-2", "[" + formas.latentes.join(", ") + "]"],
      ["Vector de hablante", "[" + formas.embedding.join(", ") + "]"],
      ["Latentes", Math.round(tiempos.latentes) + " ms"],
      ["HiFi-GAN", Math.round(tiempos.hifigan) + " ms"],
      ["DVAE frente a voz final", datos.distancia_mel === null ? "sin DVAE" :
        datos.distancia_mel.toFixed(2) + " de diferencia media en log-mel"],
    ];
    const resumen = elemento("resumen-sintesis");
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
    mensaje.textContent = (vivo ? "Voz generada: " : "Voz de una corrida guardada: ") +
      datos.duracion_s.toFixed(1) + " s. El volumen se normaliza al guardar el WAV.";
    actualizarControles();
  }

  boton.addEventListener("click", async function () {
    if (boton.disabled) return;
    limpiar();
    const version = revision;
    const id = referencia.id_corrida;
    ocupado = true;
    actualizarControles();
    mensaje.textContent = "Leyendo los estados del GPT-2 y generando la onda… la primera vez carga el vocoder.";
    const formulario = new FormData();
    formulario.append("consentimiento", "si");
    formulario.append("id_corrida", id);
    formulario.append("audio", wavReferencia, "referencia.wav");
    try {
      const respuesta = await fetch("/api/sintetizar", { method: "POST", body: formulario });
      const datos = await respuesta.json().catch(() => ({}));
      if (version !== revision) return;
      if (!respuesta.ok) {
        mensaje.textContent = (datos.errores || ["No se pudo completar la etapa 4."]).join(" ");
        return;
      }
      mostrar(datos, true);
    } catch (error) {
      if (version === revision) mensaje.textContent = "No se pudo contactar al servidor. Revisa la terminal.";
    } finally {
      if (version === revision) {
        ocupado = false;
        actualizarControles();
      }
    }
  });

  botonContraste.addEventListener("click", async function () {
    if (botonContraste.disabled) return;
    const version = revision;
    const estado = elemento("estado-contraste");
    estado.textContent = "Pasando los mismos latentes por el vocoder, sin vector de hablante…";
    botonContraste.dataset.cargando = "1";
    actualizarControles();
    try {
      const respuesta = await fetch("/api/sintetizar/contraste", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ consentimiento: "si", id_corrida: referencia.id_corrida }),
      });
      if (!respuesta.ok) {
        const datos = await respuesta.json().catch(() => ({}));
        if (version === revision) estado.textContent = (datos.errores || ["No se pudo generar el contraste."]).join(" ");
        return;
      }
      const wav = await respuesta.blob();
      if (version !== revision) return;
      const audio = elemento("audio-contraste");
      audio.src = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
      audio.classList.remove("oculto");
      estado.textContent = "Compara este audio con el de arriba: mismas palabras y mismo ritmo, distinto timbre.";
    } catch (error) {
      if (version === revision) estado.textContent = "No se pudo contactar al servidor.";
    } finally {
      delete botonContraste.dataset.cargando;
      actualizarControles();
    }
  });

  // La corrida completa (etapa 5) entrega aquí la misma respuesta que /api/sintetizar.
  window.Clonvoz = window.Clonvoz || {};
  window.Clonvoz.etapa4 = {
    mostrar: function (datos, vivo) {
      revision += 1;
      ocupado = false;
      if (vivo) tokensListos = true;
      mostrar(datos, vivo);
    },
  };

  mensaje.textContent = explicarEspera();
  actualizarControles();
})();
