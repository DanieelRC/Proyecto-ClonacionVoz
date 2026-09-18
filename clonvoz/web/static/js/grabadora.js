/*
 * Grabación desde el micrófono y codificación a WAV en el propio navegador (RF-01).
 *
 * POR QUÉ SE CODIFICA EL WAV AQUÍ Y NO EN EL SERVIDOR
 * MediaRecorder, en Chrome y Edge, entrega el audio como audio/webm con codec
 * Opus. libsndfile —la biblioteca que usa el servidor para leer audio— no sabe
 * leer eso, así que decodificarlo del lado del servidor obligaría a instalar
 * ffmpeg en Windows. En cambio aquí el navegador ya trae todo lo necesario: se
 * decodifica el blob, se remuestrea a la frecuencia de análisis y se arma un WAV
 * PCM de 16 bits.
 *
 * El efecto secundario es el que importa: el micrófono y la carga de archivo
 * entran al backend por el mismo camino, un .wav real, y RF-02 se cumple al pie
 * de la letra sin agregar ninguna dependencia de sistema.
 */

const Grabadora = (function () {
  "use strict";

  let flujo = null;
  let grabador = null;
  let trozos = [];

  /** El micrófono solo existe en contexto seguro: HTTPS o localhost. */
  function esSoportado() {
    return Boolean(
      window.isSecureContext &&
      navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia &&
      window.MediaRecorder
    );
  }

  function razonDeNoSoporte() {
    if (!window.isSecureContext) {
      return (
        "El navegador bloquea el micrófono porque la página no se abrió en un " +
        "contexto seguro. Ábrela en http://127.0.0.1:5000 (no por la IP de red)."
      );
    }
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      return "Este navegador no permite grabar audio. Usa Chrome o Edge actualizado.";
    }
    return "";
  }

  function tipoDisponible() {
    const candidatos = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];
    for (const tipo of candidatos) {
      if (window.MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(tipo)) {
        return tipo;
      }
    }
    return "";
  }

  async function iniciar() {
    // Se piden las tres mejoras de audio apagadas: la cancelación de eco, la
    // supresión de ruido y el control automático de ganancia alteran el timbre,
    // que es justo lo que la clonación necesita conservar intacto.
    flujo = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    const tipo = tipoDisponible();
    grabador = new MediaRecorder(flujo, tipo ? { mimeType: tipo } : undefined);
    trozos = [];
    grabador.ondataavailable = function (evento) {
      if (evento.data && evento.data.size > 0) {
        trozos.push(evento.data);
      }
    };
    grabador.start();
  }

  function detener() {
    return new Promise(function (resolver, rechazar) {
      if (!grabador) {
        rechazar(new Error("No hay ninguna grabación en curso."));
        return;
      }
      grabador.onstop = function () {
        const tipo = grabador.mimeType || "audio/webm";
        const grabado = new Blob(trozos, { type: tipo });
        liberar();
        resolver(grabado);
      };
      grabador.onerror = function (evento) {
        liberar();
        rechazar(evento.error || new Error("Falló la grabación."));
      };
      grabador.stop();
    });
  }

  function liberar() {
    if (flujo) {
      flujo.getTracks().forEach(function (pista) {
        pista.stop();
      });
    }
    flujo = null;
    grabador = null;
    trozos = [];
  }

  /** Decodifica cualquier audio que el navegador entienda y devuelve un AudioBuffer. */
  async function decodificar(blob) {
    const bytes = await blob.arrayBuffer();
    const Contexto = window.AudioContext || window.webkitAudioContext;
    const contexto = new Contexto();
    try {
      return await contexto.decodeAudioData(bytes);
    } finally {
      // No se espera el cierre: el buffer ya está decodificado.
      if (contexto.close) {
        contexto.close();
      }
    }
  }

  /**
   * Pasa un AudioBuffer a mono y a la frecuencia pedida.
   *
   * Se usa OfflineAudioContext en vez de interpolar a mano porque el
   * remuestreo del navegador es de buena calidad y evita el aliasing que
   * introduciría una interpolación lineal.
   */
  async function remuestrearMono(buffer, srDestino) {
    const cuadros = Math.max(1, Math.ceil(buffer.duration * srDestino));
    const offline = new OfflineAudioContext(1, cuadros, srDestino);
    const fuente = offline.createBufferSource();
    fuente.buffer = buffer;
    fuente.connect(offline.destination);
    fuente.start();
    const resultado = await offline.startRendering();
    return resultado.getChannelData(0);
  }

  /** Arma un WAV PCM de 16 bits, mono, a partir de muestras en punto flotante. */
  function codificarWav(muestras, sr) {
    const n = muestras.length;
    const buffer = new ArrayBuffer(44 + n * 2);
    const vista = new DataView(buffer);

    function texto(posicion, cadena) {
      for (let i = 0; i < cadena.length; i++) {
        vista.setUint8(posicion + i, cadena.charCodeAt(i));
      }
    }

    texto(0, "RIFF");
    vista.setUint32(4, 36 + n * 2, true);
    texto(8, "WAVE");
    texto(12, "fmt ");
    vista.setUint32(16, 16, true); // tamaño del bloque fmt
    vista.setUint16(20, 1, true); // PCM sin comprimir
    vista.setUint16(22, 1, true); // un canal
    vista.setUint32(24, sr, true);
    vista.setUint32(28, sr * 2, true); // bytes por segundo
    vista.setUint16(32, 2, true); // bytes por cuadro
    vista.setUint16(34, 16, true); // bits por muestra
    texto(36, "data");
    vista.setUint32(40, n * 2, true);

    let posicion = 44;
    for (let i = 0; i < n; i++) {
      const valor = Math.max(-1, Math.min(1, muestras[i]));
      vista.setInt16(posicion, valor < 0 ? valor * 0x8000 : valor * 0x7fff, true);
      posicion += 2;
    }

    return new Blob([buffer], { type: "audio/wav" });
  }

  /** Convierte lo grabado (webm/opus) en un .wav mono a `srDestino`. */
  async function aWav(blob, srDestino) {
    const buffer = await decodificar(blob);
    const muestras = await remuestrearMono(buffer, srDestino);
    return {
      wav: codificarWav(muestras, srDestino),
      duracion: muestras.length / srDestino,
    };
  }

  return {
    esSoportado: esSoportado,
    razonDeNoSoporte: razonDeNoSoporte,
    iniciar: iniciar,
    detener: detener,
    decodificar: decodificar,
    aWav: aWav,
  };
})();
