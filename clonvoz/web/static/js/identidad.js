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
    // Los porcentajes vienen de esta referencia, nunca de un ejemplo fijo.
    // textContent conserva el contenido como texto y evita interpretar HTML.
    const panel = document.createElement("section");
    panel.className = "explicacion-atencion";
    function texto(etiqueta, contenido) {
      const nodo = document.createElement(etiqueta);
      nodo.textContent = contenido;
      panel.appendChild(nodo);
    }
    texto("h4", "Cómo leer este resultado");
    texto("p", "La etapa 1 convierte el audio en un mel-espectrograma. En la etapa 2, el codificador procesa ese mel y el Perceiver lo resume en 32 vectores de 1024 valores. Las gráficas explican ese cálculo; no son audio generado.");
    texto("p", "Las consultas son los 32 vectores internos que el modelo va actualizando. Cada capa puede combinar información del audio codificado y de las propias consultas. En la última capa, las consultas ya han pasado por la primera y pueden contener información del audio.");
    const tabla = document.createElement("table");
    const titulo = document.createElement("caption");
    titulo.textContent = "Distribución media de la atención en esta referencia";
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
    texto("p", "Cómo se calculan: se suman los pesos destinados a todos los instantes del audio y, por separado, a las consultas; se promedian las 32 consultas y las 8 cabezas de atención y se multiplican por 100. Los dos destinos suman aproximadamente 100 % en cada capa. El redondeo puede producir pequeñas diferencias; 0.00 % puede representar un valor muy pequeño.");
    texto("p", "Estos porcentajes indican cómo se reparte la atención. No indican precisión, calidad de clonación, porcentaje de voz reconocida ni cuánto del audio se escuchó. Tampoco miden por sí solos la contribución final de cada parte al resultado.");
    texto("h4", "Por qué un mapa puede verse casi negro");
    texto("p", "Los mapas de atención solo dibujan los pesos dirigidos al audio. El eje horizontal es el tiempo y cada fila es una consulta. Oscuro significa peso bajo; claro, peso alto según la barra. El peso dirigido a las consultas no aparece en esos mapas, pero sí en la tabla.");
    (datos.resumen_atencion || []).forEach(function (capa) {
      texto("p", capa.capa + ": en este análisis destina " + porcentaje(capa.audio_porcentaje) +
        " al audio y " + porcentaje(capa.consultas_porcentaje) + " a las consultas, en promedio. " +
        (capa.consultas_porcentaje > capa.audio_porcentaje ?
          "La mayor parte va a las consultas; esa parte no colorea el mapa del audio." :
          "La mayor parte va al audio, pero puede repartirse entre muchos instantes o concentrarse en unos pocos; no tiene que iluminar todo el mapa."));
    });
    texto("p", "El color usa una escala de potencia 0.4 para hacer visibles pesos pequeños; la barra conserva los valores reales. No se inventan pesos ni se redistribuyen para llenar la imagen. Si todos son cero, aparece un mensaje. Un mapa oscuro por sí solo no demuestra un fallo ni garantiza que el resultado sea correcto.");
    texto("h4", "Qué muestran las otras vistas");
    texto("p", "Similitud entre consultas compara la dirección de los 32 vectores entre sí: 1 significa la misma dirección, 0 direcciones perpendiculares y −1 direcciones opuestas. La diagonal vale 1 para vectores no nulos. No compara personas ni mide la calidad de su voz.");
    texto("p", "En los detalles puedes ver la matriz original [32, 1024] y una versión con la media de cada dimensión restada, que destaca diferencias entre consultas. Esa resta solo sirve para visualizar: los vectores originales se conservan intactos para las siguientes etapas.");
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
      atencion_inicial: "Primera capa: pesos al audio; colores con escala de potencia 0.4.",
      atencion: "Última capa: pesos al audio. Puede atender a consultas ya procesadas.",
      similitud: "Coseno entre las 32 consultas: 1 indica la misma dirección, 0 direcciones perpendiculares y −1 opuestas. No mide identidad ni calidad de voz. Gris: consulta nula, sin coseno definido.",
      centrada: "Se resta la media de cada dimensión solo para visualizar diferencias entre consultas. No modifica la salida del modelo.",
      identidad: "Salida original [32, 1024], conservada sin cambios para las siguientes etapas. Blanco: valor cercano a cero.",
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
        pie.textContent += " Desplázate horizontalmente para recorrer el audio completo. Cada instante ocupa al menos dos píxeles, sin cambiar sus valores ni la escala de color.";
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
