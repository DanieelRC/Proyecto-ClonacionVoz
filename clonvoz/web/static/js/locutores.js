/* Etapa 7: quién habla, con la salida del Perceiver.
 * Al abrir la página se pide el reporte del entrenamiento (GET /api/locutores):
 * modelo en uso, matrices de confusión, búsqueda del KNN y archivos exportables.
 * El locutor de la referencia llega solo en cada corrida completa (evento
 * "locutor") o con el botón, que reutiliza la identidad de la etapa 2.
 */
(function () {
  "use strict";
  const elemento = (id) => document.getElementById(id);
  const boton = elemento("boton-locutor");
  const mensaje = elemento("estado-locutor");
  const porcentaje = (valor) => (100 * valor).toLocaleString("es-MX", { maximumFractionDigits: 1 }) + " %";

  let reporte = null;        // el entrenamiento, si existe
  let idConIdentidad = null; // referencia cuya etapa 2 ya está en el servidor
  let ocupado = false;
  let revision = 0;

  const autorizado = () => elemento("consentimiento").checked;

  function actualizarControles() {
    boton.disabled = !reporte || !idConIdentidad || !autorizado() || ocupado;
  }

  function celda(fila, texto, encabezado) {
    const nodo = document.createElement(encabezado ? "th" : "td");
    if (encabezado) nodo.scope = "row";
    nodo.textContent = texto;
    fila.appendChild(nodo);
    return nodo;
  }

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

  // KNN da distancias (menor = más parecido); GMM, log-verosimilitudes (mayor = más parecido).
  function puntaje(modelo, valor) {
    return modelo === "knn" ? "distancia " + valor.toFixed(3) : "log-verosimilitud " + valor.toFixed(1);
  }

  // ------------------------------------------------------------ reporte

  function mostrarReporte(datos) {
    reporte = datos;
    const elegido = datos.modelos[datos.elegido];
    elemento("modelo-en-uso").replaceChildren();
    const marca = document.createElement("strong");
    marca.textContent = "Modelo en uso: " + elegido.descripcion + ". ";
    elemento("modelo-en-uso").append(marca, "Se eligió por su exactitud en la validación cruzada: " + datos.motivo_eleccion);

    const resumen = elemento("resumen-locutores");
    resumen.replaceChildren();
    [["Locutores", datos.locutores.length], ["Audios", datos.n_audios],
     ["Validación", datos.validacion.pliegues + " pliegues"], ["Omitidos", datos.rechazados.length],
     ["Entrenado", new Date(datos.entrenado).toLocaleString("es-MX")]].forEach(function ([nombre, valor]) {
      const grupo = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = nombre;
      dd.textContent = valor;
      grupo.append(dt, dd);
      resumen.appendChild(grupo);
    });

    const modelos = elemento("tabla-modelos");
    modelos.replaceChildren();
    ["knn", "gmm"].forEach(function (clave) {
      const m = datos.modelos[clave];
      const fila = modelos.insertRow();
      if (clave === datos.elegido) fila.className = "fila--elegida";
      celda(fila, m.nombre + (clave === datos.elegido ? " · en uso" : ""), true);
      celda(fila, m.descripcion);
      celda(fila, porcentaje(m.exactitud));
      celda(fila, porcentaje(m.recall_macro));
      celda(fila, m.exactitud_pliegues.map(porcentaje).join(" · "));
      celda(fila, m.umbral === null ? "—" : (clave === "knn" ? "distancia > " + m.umbral.toFixed(3) :
        "log-verosimilitud < " + m.umbral.toFixed(1)));
    });

    pintarFigura("figura-matriz_knn", datos.graficas.matriz_knn, "Matriz de confusión del KNN: locutor real contra predicho.");
    pintarFigura("figura-matriz_gmm", datos.graficas.matriz_gmm, "Matriz de confusión del GMM: locutor real contra predicho.");
    pintarFigura("figura-proyeccion", datos.graficas.proyeccion, "Proyección PCA de los audios de entrenamiento, coloreados por locutor.");

    const busqueda = elemento("busqueda-knn");
    busqueda.replaceChildren();
    const mejor = datos.modelos.knn.configuracion;
    datos.busqueda_knn.forEach(function (f) {
      const fila = busqueda.insertRow();
      if (f.k === mejor.k && f.metrica === mejor.metrica) fila.className = "fila--elegida";
      celda(fila, String(f.k), true);
      celda(fila, f.metrica);
      celda(fila, porcentaje(f.exactitud));
      celda(fila, porcentaje(f.recall_macro));
    });

    const recall = elemento("recall-locutores");
    recall.replaceChildren();
    datos.locutores.forEach(function (locutor) {
      const fila = recall.insertRow();
      celda(fila, locutor, true);
      celda(fila, String(datos.audios_por_locutor[locutor]));
      celda(fila, porcentaje(datos.modelos.knn.recall_por_locutor[locutor] || 0));
      celda(fila, porcentaje(datos.modelos.gmm.recall_por_locutor[locutor] || 0));
    });

    const explicacion = elemento("explicacion-locutor");
    explicacion.replaceChildren();
    function parrafo(texto, fuerte) {
      const p = document.createElement("p");
      if (fuerte) {
        const b = document.createElement("strong");
        b.textContent = fuerte + " ";
        p.appendChild(b);
      }
      p.append(texto);
      explicacion.appendChild(p);
    }
    function lista(titulo, pasos) {
      parrafo("", titulo);
      const ol = document.createElement("ol");
      pasos.forEach(function (paso) {
        const li = document.createElement("li");
        li.textContent = paso;
        ol.appendChild(li);
      });
      explicacion.appendChild(ol);
    }
    lista("KNN:", datos.normalizacion.knn);
    lista("GMM:", datos.normalizacion.gmm);
    parrafo(datos.validacion.esquema + " Exactitud: aciertos entre audios de prueba. Recall (macro): " +
      "fracción reconocida de cada locutor, promediada; con el mismo número de audios por locutor coinciden.", "Validación.");
    parrafo(datos.normalizacion.desconocido + " En la validación, ese umbral habría convertido en «desconocido» " +
      describirRechazo(datos.modelos[datos.elegido]) + ".", "Locutor desconocido.");
    parrafo("La búsqueda de k y distancia se evaluó con los mismos pliegues con los que se reporta: la exactitud " +
      "del KNN elegido puede ser algo optimista.", "Advertencia.");

    const rechazados = elemento("lista-rechazados");
    rechazados.replaceChildren();
    datos.rechazados.forEach(function (r) {
      const li = document.createElement("li");
      li.textContent = r.archivo + ": " + r.motivo + ".";
      rechazados.appendChild(li);
    });
    elemento("detalle-rechazados").classList.toggle("oculto", !datos.rechazados.length);

    const exportables = elemento("exportables");
    exportables.replaceChildren();
    const todo = document.createElement("a");
    todo.className = "boton boton--primario";
    todo.href = "/api/locutores/exportar.zip";
    todo.textContent = "Descargar todo (.zip)";
    exportables.appendChild(todo);
    datos.exportados.forEach(function (nombre) {
      const enlace = document.createElement("a");
      enlace.className = "boton boton--discreto";
      enlace.href = "/api/locutores/exportar/" + encodeURIComponent(nombre);
      enlace.textContent = nombre;
      enlace.setAttribute("download", nombre);
      exportables.appendChild(enlace);
    });

    elemento("reporte-locutores").classList.remove("oculto");
  }

  function describirRechazo(modelo) {
    if (modelo.aciertos_rechazados === null) return "ningún audio";
    let texto = porcentaje(modelo.aciertos_rechazados) + " de los aciertos";
    if (modelo.errores_rechazados !== null) texto += " y " + porcentaje(modelo.errores_rechazados) + " de los errores";
    return texto;
  }

  async function cargarReporte() {
    const version = ++revision;
    elemento("modelo-en-uso").textContent = "Buscando el reconocedor entrenado…";
    try {
      const respuesta = await fetch("/api/locutores", { cache: "no-store" });
      const datos = await respuesta.json();
      if (version !== revision) return;
      if (!datos.hay_modelo) {
        reporte = null;
        elemento("reporte-locutores").classList.add("oculto");
        elemento("modelo-en-uso").textContent = "Todavía no hay un reconocedor entrenado. Ejecuta " +
          "python -m clonvoz.cli_locutores --consentimiento entrenar <carpeta> y pulsa Recargar reporte. " +
          "Mientras tanto la corrida sigue y este bloque aparece como omitido.";
      } else {
        mostrarReporte(datos);
      }
    } catch (error) {
      if (version === revision)
        elemento("modelo-en-uso").textContent = "Sin servidor no hay reporte; una corrida grabada sí muestra su locutor.";
    }
    actualizarControles();
  }

  // ---------------------------------------------------------- resultado

  function limpiarResultado() {
    elemento("resultado-locutor").classList.add("oculto");
    mensaje.textContent = "";
  }

  function mostrar(datos) {
    const veredicto = elemento("veredicto");
    veredicto.classList.toggle("veredicto--desconocido", datos.desconocido);
    elemento("veredicto-etiqueta").textContent = (datos.desconocido ? "Sin coincidencia según " : "Identificado por ") +
      datos.descripcion;
    elemento("veredicto-nombre").textContent = datos.locutor;
    let detalle = puntaje(datos.modelo, datos.puntaje);
    if (datos.umbral !== null) detalle += " · umbral " + (datos.modelo === "knn" ? datos.umbral.toFixed(3) : datos.umbral.toFixed(1));
    if (datos.desconocido) detalle += " · el más parecido es " + datos.mas_parecido;
    detalle += ". El " + datos.otro.nombre_modelo + " dice: " + datos.otro.locutor +
      (datos.otro.locutor === "Desconocido" ? " (más parecido: " + datos.otro.mas_parecido + ")" : "") + ".";
    elemento("veredicto-detalle").textContent = detalle;

    const cuerpo = elemento("candidatos-locutor");
    cuerpo.replaceChildren();
    [datos, datos.otro].forEach(function (resultado) {
      const fila = cuerpo.insertRow();
      celda(fila, resultado.nombre_modelo + (resultado === datos ? " · en uso" : ""), true);
      for (let i = 0; i < 3; i += 1) {
        const c = resultado.candidatos[i];
        celda(fila, c ? c.locutor + " (" + (resultado.modelo === "knn" ? c.puntaje.toFixed(3) : c.puntaje.toFixed(1)) + ")" : "—");
      }
    });
    pintarFigura("figura-locutor-proyeccion", datos.graficas.proyeccion,
      "Proyección PCA de los locutores conocidos con este audio marcado como estrella.");
    elemento("resultado-locutor").classList.remove("oculto");
    mensaje.textContent = (datos.archivo ? datos.archivo + ": " : "") + "identificado en " + datos.ms + " ms con el modelo entrenado el " +
      new Date(datos.entrenado).toLocaleString("es-MX");
  }

  boton.addEventListener("click", async function () {
    if (boton.disabled) return;
    ocupado = true;
    actualizarControles();
    const id = idConIdentidad;
    mensaje.textContent = "Comparando la salida del Perceiver con los locutores conocidos…";
    try {
      const respuesta = await fetch("/api/locutor", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_corrida: id, consentimiento: "si" }),
      });
      const datos = await respuesta.json();
      if (id !== idConIdentidad) return;
      if (!respuesta.ok) {
        if (respuesta.status === 410) idConIdentidad = null;
        mensaje.textContent = (datos.errores || ["No se pudo identificar al locutor."]).join(" ");
        return;
      }
      mostrar(datos);
    } catch (error) {
      mensaje.textContent = "No se pudo contactar al servidor.";
    } finally {
      ocupado = false;
      actualizarControles();
    }
  });

  elemento("recargar-locutores").addEventListener("click", cargarReporte);

  function olvidar() {
    idConIdentidad = null;
    limpiarResultado();
    actualizarControles();
  }
  document.addEventListener("referencia-cambiada", olvidar);
  document.addEventListener("etapa1-iniciada", olvidar);
  document.addEventListener("etapa2-completada", function (evento) {
    idConIdentidad = evento.detail?.id_corrida || null;
    actualizarControles();
  });
  elemento("consentimiento").addEventListener("change", function () {
    if (!autorizado()) olvidar();
    actualizarControles();
  });

  // La corrida completa (en vivo o grabada) entrega aquí el evento "locutor".
  window.Clonvoz = window.Clonvoz || {};
  window.Clonvoz.locutor = { mostrar, limpiar: limpiarResultado };

  cargarReporte();
})();
