/* Etapa 1 -> memoria temporal del servidor -> etapa 2.
 * Aquí solo se envían identificadores de análisis terminados, nunca archivos.
 * Cambiar la referencia invalida etapa 2; conservar una comparación es explícito.
 */
(function () {
  "use strict";
  const elemento = (id) => document.getElementById(id);
  const boton = elemento("boton-identidad");
  const conservar = elemento("conservar-comparacion");
  const mensaje = elemento("estado-identidad");
  const resultados = elemento("resultados-identidad");
  let referencia = null;
  let comparacion = null;
  let solicitud = null;
  let revision = 0;

  function descartar(datos) {
    if (!datos) return;
    fetch("/api/descartar", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_corrida: datos.id_corrida }), keepalive: true,
    }).catch(function () { /* El servidor también elimina por caducidad. */ });
  }

  function actualizarControles() {
    const autorizado = elemento("consentimiento").checked;
    boton.disabled = !referencia || !autorizado || !!solicitud;
    conservar.disabled = !referencia || !autorizado || !!solicitud;
    elemento("referencia-comparacion").textContent = comparacion ?
      "Conservada: " + comparacion.archivo : "Sin referencia conservada.";
  }

  function invalidarResultados() {
    // Cancelar la espera no detiene Python, pero la revisión impide mostrar
    // una respuesta antigua sobre una selección nueva.
    revision += 1;
    if (solicitud) solicitud.abort();
    solicitud = null;
    resultados.replaceChildren();
    mensaje.textContent = "";
  }

  function invalidarReferencia() {
    invalidarResultados();
    // Un mel conservado para comparar debe sobrevivir al cambio de referencia.
    if (referencia && referencia.id_corrida !== comparacion?.id_corrida) descartar(referencia);
    referencia = null;
    mensaje.textContent = "Completa la etapa 1 para habilitar la etapa 2.";
    actualizarControles();
  }
  document.addEventListener("referencia-cambiada", invalidarReferencia);
  document.addEventListener("etapa1-iniciada", invalidarReferencia);
  document.addEventListener("etapa1-completada", function (evento) {
    referencia = evento.detail;
    invalidarResultados();
    mensaje.textContent = "Mel de etapa 1 listo. Ya puedes ejecutar la etapa 2.";
    actualizarControles();
  });
  elemento("consentimiento").addEventListener("change", function () {
    if (!elemento("consentimiento").checked) {
      descartar(comparacion);
      comparacion = null;
      invalidarReferencia();
    }
    actualizarControles();
  });

  conservar.addEventListener("click", function () {
    if (!referencia) return;
    if (comparacion && comparacion.id_corrida !== referencia.id_corrida) descartar(comparacion);
    comparacion = referencia;
    invalidarResultados();
    mensaje.textContent = "Análisis conservado. Carga otra referencia arriba y ejecuta su etapa 1.";
    actualizarControles();
  });
  elemento("quitar-comparacion").addEventListener("click", function () {
    if (comparacion && comparacion.id_corrida !== referencia?.id_corrida) descartar(comparacion);
    comparacion = null;
    invalidarResultados();
    actualizarControles();
  });
  // Al abandonar la página se intenta liberar ambos mels. La caducidad del
  // servidor cubre cierres bruscos o pérdida de conexión que impidan este envío.
  window.addEventListener("pagehide", function () {
    descartar(referencia);
    descartar(comparacion);
  });

  function explicarAtencion(datos, columna) {
    // Plegado: la tabla con los porcentajes de esta referencia y tres frases.
    // textContent conserva el contenido como texto y evita interpretar HTML.
    const panel = document.createElement("details");
    panel.className = "detalle explicacion-atencion";
    const resumen = document.createElement("summary");
    resumen.textContent = "Cómo leer este resultado";
    panel.appendChild(resumen);
    function texto(contenido) {
      const nodo = document.createElement("p");
      nodo.textContent = contenido;
      panel.appendChild(nodo);
    }
    const tabla = document.createElement("table");
    const titulo = document.createElement("caption");
    titulo.textContent = "A dónde va la atención de las 32 consultas, en promedio";
    tabla.appendChild(titulo);
    const cabecera = tabla.createTHead().insertRow();
    ["Capa", "Al audio", "A las consultas"].forEach(function (nombre) {
      const celda = document.createElement("th");
      celda.scope = "col";
      celda.textContent = nombre;
      cabecera.appendChild(celda);
    });
    const cuerpo = tabla.createTBody();
    const porcentaje = (valor) => valor.toLocaleString("es-MX", {minimumFractionDigits: 2, maximumFractionDigits: 2}) + " %";
    (datos.resumen_atencion || []).forEach(function (capa) {
      const fila = cuerpo.insertRow();
      const nombre = document.createElement("th");
      nombre.scope = "row";
      nombre.textContent = capa.capa;
      fila.appendChild(nombre);
      fila.insertCell().textContent = porcentaje(capa.audio_porcentaje);
      fila.insertCell().textContent = porcentaje(capa.consultas_porcentaje);
    });
    panel.appendChild(tabla);
    texto("Los mapas solo dibujan el peso que va al audio (tiempo en horizontal, una consulta por fila); " +
      "el que va a las propias consultas está en la tabla. Por eso un mapa puede verse casi oscuro.");
    texto("El color usa una potencia 0.4 para revelar pesos pequeños; los valores no se renormalizan.");
    texto("Repartir atención no mide calidad de clonación ni identifica a una persona.");
    columna.appendChild(panel);
  }

  function mostrarResultado(datos) {
    // 4a. Cada referencia ocupa un artículo. textContent muestra nombres/textos
    // como texto plano; un nombre de archivo no puede convertirse en código HTML.
    const columna = document.createElement("article");
    const titulo = document.createElement("h3");
    titulo.textContent = datos.archivo;
    columna.appendChild(titulo);
    explicarAtencion(datos, columna);
    // La vista principal explica relaciones; los detalles conservan los valores.
    const descripciones = {
      atencion_inicial: "Primera capa: atención de cada consulta al audio.",
      atencion: "Última capa: atención de cada consulta al audio.",
      similitud: "Coseno entre las 32 consultas: 1 misma dirección, −1 opuesta. No compara personas.",
      centrada: "Diferencias entre consultas (se resta la media solo para verlas).",
      identidad: "Salida original [32, 1024] que usan las etapas siguientes.",
    };
    ["atencion_inicial", "atencion", "similitud", "centrada", "identidad"].forEach(function (nombre) {
      // 4b. Dos imágenes del mismo mapa: CSS elige la variante clara u oscura.
      // Ambas llegan ya calculadas; cambiar tema no vuelve a consultar al modelo.
      const figura = document.createElement("figure");
      figura.className = "figura";
      const lienzo = document.createElement("div");
      lienzo.className = "figura__lienzo";
      if (nombre.startsWith("atencion")) {
        // Conserva el ancho del PNG: reducirlo volvería a ocultar los picos.
        lienzo.classList.add("mapa-atencion-desplazable");
        lienzo.tabIndex = 0;
        lienzo.setAttribute("role", "region");
        lienzo.setAttribute("aria-label", "Mapa de atención con desplazamiento horizontal");
      }
      ["claro", "oscuro"].forEach(function (tema) {
        const imagen = document.createElement("img");
        imagen.className = "figura__imagen figura__imagen--" + tema;
        imagen.src = datos.graficas[nombre][tema];
        imagen.alt = descripciones[nombre];
        if (tema === "oscuro") imagen.setAttribute("aria-hidden", "true");
        // El lector de pantalla recibe una sola descripción por mapa, sin duplicarla.
        lienzo.appendChild(imagen);
      });
      const pie = document.createElement("figcaption");
      pie.textContent = descripciones[nombre];
      if (nombre.startsWith("atencion")) {
        pie.textContent += " Desplázate en horizontal para ver todo el audio.";
      }
      figura.append(lienzo, pie);
      if (nombre === "identidad" || nombre === "centrada") {
        const plegable = document.createElement("details");
        const resumen = document.createElement("summary");
        resumen.textContent = nombre === "identidad" ? "Ver matriz original" : "Ver diferencias entre consultas";
        plegable.append(resumen, figura);
        columna.appendChild(plegable);
      } else {
        columna.appendChild(figura);
      }
    });
    const detalle = document.createElement("p");
    // 4c. Los intervalos explican qué parte del audio corresponde a cada fragmento.
    // toFixed limita decimales solo en pantalla, sin cambiar los datos originales.
    detalle.className = "ayuda";
    detalle.textContent = "Audio analizado: " + datos.fragmentos.map((f) =>
      f.inicio_s.toFixed(2) + "–" + f.fin_s.toFixed(2) + " s").join("; ") +
      ". Dispositivo: " + datos.dispositivo + ".";
    columna.appendChild(detalle);
    const tiempos = document.createElement("p");
    // Object.entries permite recorrer cada pareja nombre/tiempo del resumen.
    tiempos.className = "ayuda";
    tiempos.textContent = "Tiempos (ms): " + Object.entries(datos.tiempos_ms).map(([k, v]) => k + ": " + v).join(" · ");
    columna.appendChild(tiempos);
    const id = document.createElement("p");
    id.className = "identificador";
    id.textContent = "Corrida " + datos.id_corrida;
    columna.appendChild(id);
    datos.advertencias.forEach(function (texto) {
      const aviso = document.createElement("p");
      aviso.textContent = texto;
      columna.appendChild(aviso);
    });
    resultados.appendChild(columna);
  }

  // La corrida completa (etapa 5) entrega aquí la misma respuesta que /api/identidad.
  window.Clonvoz = window.Clonvoz || {};
  window.Clonvoz.etapa2 = {
    mostrar: function (datos, enVivo) {
      invalidarResultados();
      datos.resultados.forEach(mostrarResultado);
      resultados.classList.toggle("comparacion-identidad--doble", datos.resultados.length === 2);
      mensaje.textContent = enVivo ? "Etapa 2 terminada dentro de la corrida completa." :
        "Etapa 2 de una corrida guardada.";
      if (enVivo && referencia)
        document.dispatchEvent(new CustomEvent("etapa2-completada", { detail: { id_corrida: referencia.id_corrida } }));
      actualizarControles();
    },
  };

  boton.addEventListener("click", async function () {
    if (!referencia || !elemento("consentimiento").checked) return;
    invalidarResultados();
    const version = revision;
    solicitud = new AbortController();
    actualizarControles();
    mensaje.textContent = "Procesando el mel de etapa 1 con el Perceiver…";
    const cuerpo = { id_corrida: referencia.id_corrida, consentimiento: "si" };
    if (comparacion && comparacion.id_corrida !== referencia.id_corrida)
      cuerpo.comparacion_id = comparacion.id_corrida;
    try {
      const respuesta = await fetch("/api/identidad", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cuerpo), signal: solicitud.signal,
      });
      const datos = await respuesta.json();
      if (version !== revision) return;
      if (!respuesta.ok) {
        if (respuesta.status === 410) {
          descartar(referencia);
          descartar(comparacion);
          referencia = null;
          comparacion = null;
        }
        mensaje.textContent = (datos.errores || ["No se pudo completar la etapa 2."]).join(" ");
        return;
      }
      datos.resultados.forEach(mostrarResultado);
      resultados.classList.toggle("comparacion-identidad--doble", datos.resultados.length === 2);
      document.dispatchEvent(new CustomEvent("etapa2-completada", { detail: { id_corrida: referencia.id_corrida } }));
      mensaje.textContent = "Etapa 2 terminada usando el mel de etapa 1. Carga del modelo: " +
        datos.carga_modelo_ms + " ms. No se volvió a calcular el mel.";
    } catch (error) {
      if (error.name !== "AbortError" && version === revision)
        mensaje.textContent = "No se pudo contactar al servidor. Revisa la terminal.";
    } finally {
      if (version === revision) {
        solicitud = null;
        actualizarControles();
      }
    }
  });
  actualizarControles();
})();
