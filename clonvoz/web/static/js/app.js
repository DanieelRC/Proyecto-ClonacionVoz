/*
 * Interfaz de la etapa 1: consentimiento, captura, validación y presentación.
 *
 * El navegador valida antes de enviar para dar respuesta inmediata, pero la
 * validación que manda es la del servidor (RF-03): estas reglas son una cortesía,
 * no la autoridad.
 */

(function () {
  "use strict";

  const CONFIGURACION = JSON.parse(document.getElementById("configuracion").textContent);

  const FIGURAS = {
    forma_de_onda: "Forma de onda del audio: amplitud contra tiempo.",
    espectrograma: "Espectrograma STFT: mapa de calor de frecuencia contra tiempo.",
    mel: "Mel-espectrograma de " + CONFIGURACION.nMels +
      " canales: mapa de calor de canal Mel contra tiempo.",
    f0: "Contorno de frecuencia fundamental en hercios, junto a la forma de onda, " +
      "sobre el mismo eje de tiempo.",
  };

  const NOMBRES_DE_PASO = {
    carga: "Carga y remuestreo",
    forma_de_onda: "Forma de onda",
    espectrograma: "Espectrograma STFT",
    mel: "Mel-espectrograma",
    f0: "Contorno de F0",
  };

  const elemento = function (id) {
    return document.getElementById(id);
  };

  const estado = {
    wav: null,
    nombre: "",
    duracion: 0,
    urlReproductor: null,
    grabando: false,
    temporizador: null,
    inicioGrabacion: 0,
  };

  // ------------------------------------------------------------------ tema

  function aplicarTema(tema, recordar) {
    document.documentElement.setAttribute("data-tema", tema);
    elemento("etiqueta-tema").textContent = tema === "oscuro" ? "Modo claro" : "Modo oscuro";
    if (recordar) {
      try {
        localStorage.setItem("tema", tema);
      } catch (e) {
        /* Ventana privada o almacenamiento bloqueado: el tema sigue funcionando,
           solo no se recuerda para la próxima visita. */
      }
    }
  }

  function prepararTema() {
    aplicarTema(document.documentElement.getAttribute("data-tema") || "claro", false);

    elemento("conmutador-tema").addEventListener("click", function () {
      const actual = document.documentElement.getAttribute("data-tema");
      aplicarTema(actual === "oscuro" ? "claro" : "oscuro", true);
    });

    // Si nadie eligió tema a mano, se sigue al sistema operativo.
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (evento) {
        let elegido = null;
        try {
          elegido = localStorage.getItem("tema");
        } catch (e) {
          elegido = null;
        }
        if (elegido !== "claro" && elegido !== "oscuro") {
          aplicarTema(evento.matches ? "oscuro" : "claro", false);
        }
      });
    }
  }

  // -------------------------------------------------------------- mensajes

  function limpiarMensajes() {
    elemento("mensajes").textContent = "";
  }

  function mostrarMensajes(textos, clase) {
    const contenedor = elemento("mensajes");
    (textos || []).forEach(function (texto) {
      const fila = document.createElement("div");
      fila.className = "mensaje " + clase;

      const icono = document.createElement("span");
      icono.className = "mensaje__icono";
      icono.setAttribute("aria-hidden", "true");
      icono.textContent = clase === "mensaje--error" ? "✕" : "!";

      const cuerpo = document.createElement("span");
      // El texto dice siempre de qué se trata: el color nunca es la única señal.
      cuerpo.textContent = (clase === "mensaje--error" ? "Error: " : "Advertencia: ") + texto;

      fila.appendChild(icono);
      fila.appendChild(cuerpo);
      contenedor.appendChild(fila);
    });
  }

  const mostrarErrores = function (textos) {
    mostrarMensajes(textos, "mensaje--error");
  };
  const mostrarAvisos = function (textos) {
    mostrarMensajes(textos, "mensaje--aviso");
  };

  // ---------------------------------------------------- consentimiento (RNF-12)

  function prepararConsentimiento() {
    const acepto = elemento("acepto");

    const sincronizar = function () {
      const permitido = acepto.checked;
      elemento("tarjeta-captura").setAttribute("aria-disabled", permitido ? "false" : "true");
      elemento("archivo").disabled = !permitido;
      elemento("boton-grabar").disabled = !permitido || !Grabadora.esSoportado();
      elemento("boton-analizar").disabled = !permitido || !estado.wav;

      if (permitido && !Grabadora.esSoportado()) {
        limpiarMensajes();
        mostrarAvisos([Grabadora.razonDeNoSoporte() + " Puedes subir un archivo .wav."]);
      }
    };

    acepto.addEventListener("change", sincronizar);
    sincronizar();
  }

  // ------------------------------------------------------- validación local

  function validarLocalmente(duracion) {
    const errores = [];
    const avisos = [];

    if (duracion < CONFIGURACION.duracionMinima) {
      errores.push(
        "El audio dura " + duracion.toFixed(1) + " s y se necesitan al menos " +
        CONFIGURACION.duracionMinima.toFixed(0) + " s."
      );
    } else if (duracion > CONFIGURACION.duracionMaxima) {
      errores.push(
        "El audio dura " + duracion.toFixed(1) + " s y el máximo son " +
        CONFIGURACION.duracionMaxima.toFixed(0) + " s."
      );
    } else if (duracion < CONFIGURACION.duracionRecomendadaMinima) {
      avisos.push(
        "El audio dura " + duracion.toFixed(1) + " s. Se puede analizar, pero la " +
        "clonación funciona mejor con referencias de " +
        CONFIGURACION.duracionRecomendadaMinima.toFixed(0) + " a " +
        CONFIGURACION.duracionRecomendadaMaxima.toFixed(0) + " s."
      );
    }
    return { errores: errores, avisos: avisos };
  }

  function fijarReferencia(wav, nombre, duracion) {
    estado.wav = wav;
    estado.nombre = nombre;
    estado.duracion = duracion;

    if (estado.urlReproductor) {
      URL.revokeObjectURL(estado.urlReproductor);
    }
    estado.urlReproductor = URL.createObjectURL(wav);

    elemento("audio-referencia").src = estado.urlReproductor;
    elemento("nombre-archivo").textContent =
      nombre + " · " + duracion.toFixed(1) + " s · " + Math.round(wav.size / 1024) + " KB";
    elemento("reproductor").classList.remove("oculto");
    elemento("boton-analizar").disabled = !elemento("acepto").checked;
  }

  // --------------------------------------------------------- carga de archivo

  async function recibirArchivo(archivo) {
    limpiarMensajes();

    if (!archivo) {
      return;
    }
    if (!/\.wav$/i.test(archivo.name)) {
      mostrarErrores([
        "Solo se aceptan archivos .wav. Convierte el audio antes de subirlo " +
        "(cambiarle la extensión al nombre no lo convierte).",
      ]);
      return;
    }
    if (archivo.size > CONFIGURACION.tamanoMaximoMb * 1024 * 1024) {
      mostrarErrores([
        "El archivo pesa más de " + CONFIGURACION.tamanoMaximoMb + " MB.",
      ]);
      return;
    }

    let buffer;
    try {
      buffer = await Grabadora.decodificar(archivo);
    } catch (e) {
      mostrarErrores([
        "No se pudo leer el archivo como audio. Puede estar dañado, incompleto, " +
        "o no ser realmente un .wav.",
      ]);
      return;
    }

    const revision = validarLocalmente(buffer.duration);
    if (revision.errores.length) {
      mostrarErrores(revision.errores);
      return;
    }
    mostrarAvisos(revision.avisos);
    fijarReferencia(archivo, archivo.name, buffer.duration);
  }

  function prepararCarga() {
    const entrada = elemento("archivo");
    const zona = elemento("zona-carga");

    entrada.addEventListener("change", function () {
      recibirArchivo(entrada.files && entrada.files[0]);
    });

    ["dragenter", "dragover"].forEach(function (evento) {
      zona.addEventListener(evento, function (e) {
        e.preventDefault();
        if (!entrada.disabled) {
          zona.classList.add("zona-carga--activa");
        }
      });
    });

    ["dragleave", "drop"].forEach(function (evento) {
      zona.addEventListener(evento, function () {
        zona.classList.remove("zona-carga--activa");
      });
    });

    zona.addEventListener("drop", function (e) {
      e.preventDefault();
      if (entrada.disabled) {
        return;
      }
      const archivos = e.dataTransfer && e.dataTransfer.files;
      if (archivos && archivos.length) {
        recibirArchivo(archivos[0]);
      }
    });
  }

  // -------------------------------------------------------------- grabación

  function formatearReloj(segundos) {
    const minutos = Math.floor(segundos / 60);
    const resto = Math.floor(segundos % 60);
    return String(minutos).padStart(2, "0") + ":" + String(resto).padStart(2, "0");
  }

  function iniciarCronometro() {
    estado.inicioGrabacion = Date.now();
    estado.temporizador = window.setInterval(function () {
      const transcurrido = (Date.now() - estado.inicioGrabacion) / 1000;
      elemento("cronometro").textContent = formatearReloj(transcurrido);
      // Se corta sola al llegar al máximo, en vez de dejar grabar algo que el
      // servidor va a rechazar.
      if (transcurrido >= CONFIGURACION.duracionMaxima) {
        detenerGrabacion();
      }
    }, 200);
  }

  function detenerCronometro() {
    if (estado.temporizador) {
      window.clearInterval(estado.temporizador);
      estado.temporizador = null;
    }
  }

  async function iniciarGrabacion() {
    limpiarMensajes();
    try {
      await Grabadora.iniciar();
    } catch (e) {
      mostrarErrores([
        "No se pudo acceder al micrófono. Revisa que le hayas dado permiso al " +
        "navegador y que ninguna otra aplicación lo esté usando.",
      ]);
      return;
    }
    estado.grabando = true;
    const boton = elemento("boton-grabar");
    boton.classList.add("boton--grabando");
    elemento("etiqueta-grabar").textContent = "Detener";
    elemento("cronometro").textContent = "00:00";
    iniciarCronometro();
  }

  async function detenerGrabacion() {
    if (!estado.grabando) {
      return;
    }
    estado.grabando = false;
    detenerCronometro();

    const boton = elemento("boton-grabar");
    boton.classList.remove("boton--grabando");
    boton.disabled = true;
    elemento("etiqueta-grabar").textContent = "Procesando…";

    try {
      const grabado = await Grabadora.detener();
      const resultado = await Grabadora.aWav(grabado, CONFIGURACION.sr);

      const revision = validarLocalmente(resultado.duracion);
      if (revision.errores.length) {
        mostrarErrores(revision.errores);
      } else {
        mostrarAvisos(revision.avisos);
        fijarReferencia(resultado.wav, "grabacion.wav", resultado.duracion);
      }
    } catch (e) {
      mostrarErrores(["No se pudo procesar la grabación. Inténtalo de nuevo."]);
    } finally {
      boton.disabled = !elemento("acepto").checked;
      elemento("etiqueta-grabar").textContent = "Empezar a grabar";
    }
  }

  function prepararGrabacion() {
    elemento("boton-grabar").addEventListener("click", function () {
      if (estado.grabando) {
        detenerGrabacion();
      } else {
        iniciarGrabacion();
      }
    });
  }

  // ---------------------------------------------------------------- análisis

  function pintarFiguras(graficas, formas) {
    Object.keys(FIGURAS).forEach(function (nombre) {
      const figura = elemento("figura-" + nombre);
      if (!figura || !graficas[nombre]) {
        return;
      }

      const lienzo = figura.querySelector(".figura__lienzo");
      lienzo.textContent = "";

      ["claro", "oscuro"].forEach(function (tema) {
        const imagen = document.createElement("img");
        imagen.className = "figura__imagen figura__imagen--" + tema;
        imagen.src = graficas[nombre][tema];
        // Solo una de las dos variantes se describe, para que un lector de
        // pantalla no anuncie la misma figura dos veces.
        if (tema === "claro") {
          imagen.alt = FIGURAS[nombre];
        } else {
          imagen.alt = "";
          imagen.setAttribute("aria-hidden", "true");
        }
        lienzo.appendChild(imagen);
      });

      const forma = figura.querySelector("[data-forma]");
      if (forma && formas[nombre]) {
        forma.textContent = "tensor [" + formas[nombre].join(", ") + "]";
      }
    });
  }

  function pintarResumen(datos) {
    const meta = datos.metadatos;
    const est = datos.estadisticas;

    const filas = [
      ["Duración", meta.duracion_s.toFixed(2) + " s"],
      ["Muestreo", meta.sr_original + " → " + meta.sr_trabajo + " Hz"],
      ["Frames (T)", String(meta.n_frames)],
      ["Mel", "[" + datos.formas.mel.join(", ") + "]"],
      [
        "F0 media",
        est.f0_media_hz === null
          ? "sin voz detectada"
          : est.f0_media_hz.toFixed(0) + " Hz (" +
            est.f0_min_hz.toFixed(0) + "–" + est.f0_max_hz.toFixed(0) + ")",
      ],
      ["Frames con voz", est.porcentaje_con_voz.toFixed(1) + " %"],
      ["Pico / RMS", est.pico.toFixed(3) + " / " + est.rms.toFixed(4)],
      ["Estimador de F0", meta.metodo_f0],
    ];

    const lista = elemento("resumen");
    lista.textContent = "";
    filas.forEach(function (fila) {
      const grupo = document.createElement("div");
      const termino = document.createElement("dt");
      termino.textContent = fila[0];
      const definicion = document.createElement("dd");
      definicion.textContent = fila[1];
      grupo.appendChild(termino);
      grupo.appendChild(definicion);
      lista.appendChild(grupo);
    });

    // RF-20: tiempos por paso, para saber qué cuesta de verdad cada cálculo.
    const cuerpo = elemento("tiempos");
    cuerpo.textContent = "";
    let total = 0;
    Object.keys(datos.tiempos_ms).forEach(function (paso) {
      const ms = datos.tiempos_ms[paso];
      total += ms;
      const fila = document.createElement("tr");
      const nombre = document.createElement("td");
      nombre.textContent = NOMBRES_DE_PASO[paso] || paso;
      const valor = document.createElement("td");
      valor.textContent = ms.toFixed(1);
      fila.appendChild(nombre);
      fila.appendChild(valor);
      cuerpo.appendChild(fila);
    });

    const filaTotal = document.createElement("tr");
    const etiquetaTotal = document.createElement("td");
    etiquetaTotal.innerHTML = "<strong>Total</strong>";
    const valorTotal = document.createElement("td");
    valorTotal.innerHTML = "<strong>" + total.toFixed(1) + "</strong>";
    filaTotal.appendChild(etiquetaTotal);
    filaTotal.appendChild(valorTotal);
    cuerpo.appendChild(filaTotal);

    elemento("identificador").textContent = "Identificador de corrida: " + datos.id_corrida;
  }

  async function analizar() {
    if (!estado.wav) {
      return;
    }
    limpiarMensajes();

    const boton = elemento("boton-analizar");
    boton.disabled = true;
    elemento("cargando").classList.remove("oculto");

    const cuerpo = new FormData();
    cuerpo.append("audio", estado.wav, estado.nombre || "referencia.wav");
    cuerpo.append("metodo_f0", "pyin");

    try {
      const respuesta = await fetch("/api/analizar", { method: "POST", body: cuerpo });
      const datos = await respuesta.json();

      if (!respuesta.ok) {
        mostrarErrores(datos.errores || ["No se pudo analizar el audio."]);
        return;
      }

      mostrarAvisos(datos.advertencias);
      pintarFiguras(datos.graficas, datos.formas);
      pintarResumen(datos);
      elemento("resultados").classList.remove("oculto");
      elemento("resultados").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      mostrarErrores([
        "No se pudo contactar al servidor. Revisa que siga corriendo en la terminal.",
      ]);
    } finally {
      elemento("cargando").classList.add("oculto");
      boton.disabled = false;
    }
  }

  // ------------------------------------------------------------------ arranque

  prepararTema();
  prepararConsentimiento();
  prepararCarga();
  prepararGrabacion();
  elemento("boton-analizar").addEventListener("click", analizar);
})();
