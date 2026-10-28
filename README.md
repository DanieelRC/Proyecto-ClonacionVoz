## Requisitos del Sistema

- **Sistema Operativo:** Windows 10/11 (probado en Windows 64-bit).
- **Versión de Python:** **Python 3.10 o 3.11** (probado con 3.10.9 y 3.11.9).
- **Audio de referencia:** Archivo en formato `.wav` (por ejemplo, `muestra_hablante.wav`).

## Instalación y Configuración

### 1. Crear el entorno virtual

Desde la terminal en la raíz del proyecto:

```powershell
python -m venv .venv
```

### 2. Activar el entorno virtual

- **PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **CMD (Símbolo del sistema):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

### 3. Instalar las dependencias exactas

Un solo archivo cubre las etapas 1 a 7:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Instala la variante CPU de PyTorch 2.8.0. PyTorch se queda en esa versión por
una razón concreta: desde la 2.9, coqui-tts exige además el paquete `torchcodec`
y aborta la importación si no está (`TTS/__init__.py`).

Si el entorno viene de una instalación anterior y `import TTS` falla por
Coqpit, quita el paquete original y reinstala el fork:

```powershell
pip uninstall -y coqpit
pip install --force-reinstall --no-deps coqpit-config==0.2.5
```

---

## Etapa 1 — Preprocesamiento de audio y características visuales

Primer prototipo funcional del proyecto. Carga o graba un audio y extrae de él
cuatro representaciones, **sin XTTS-v2, sin GPU y sin ninguna red neuronal**:

1. **Forma de onda** — amplitud contra tiempo.
2. **Espectrograma STFT** — frecuencia contra tiempo, por Transformada de Fourier
   de Tiempo Corto.
3. **Mel-espectrograma de 80 canales** — con los parámetros exactos de XTTS-v2,
   para que la etapa 2 lo pueda consumir tal cual.
4. **Contorno de F0** — la entonación a lo largo del audio.

### Ejecutar la aplicación web

```powershell
python app.py
```

Luego abre **http://127.0.0.1:5000** en Chrome o Edge.

> **Usa `127.0.0.1`, no la IP de red de la máquina.** El navegador solo permite
> usar el micrófono en un "contexto seguro": HTTPS o localhost. Abierta por
> `192.168.x.x` la página carga, pero el micrófono queda bloqueado.

El audio se procesa en memoria y no se guarda en ningún archivo del servidor.

### Ejecutar solo el procesamiento, sin servidor

```powershell
python -m clonvoz.cli audios/muestra_hablante.wav --salida salidas/ --guardar-mel
```

Escribe las cuatro imágenes, un `caracteristicas.json` con formas, estadísticas y
tiempos, y —con `--guardar-mel`— un `mel.npy` de `[80, T]` listo para la etapa 2.
Opciones: `--tema claro|oscuro|ambos`, `--metodo-f0 pyin|yin`.

### Verificar la compatibilidad con la etapa 2

```powershell
python scripts/verificar_compatibilidad_mel.py audios/muestra_hablante.wav
```

Compara el mel-espectrograma contra `wav_to_mel_cloning` de XTTS-v2 y
falla si la diferencia supera `1e-4`. Vale la pena correrlo cada vez que se
toquen los parámetros: si el mel no coincide, el error no aparece en esta etapa
sino en la etapa 2, disfrazado de "el modelo no funciona".

### Estructura del código

```
app.py                              lanza el servidor web
clonvoz/
  config.py                         parámetros de XTTS-v2, en un solo lugar
  audio.py                          carga y validación del .wav
  caracteristicas.py                las cuatro características (DSP puro)
  graficas.py                       render de las figuras a PNG
  analisis.py                       orquestación, tiempos e identificador de corrida
  cli.py                            línea de comandos
  web/servidor.py                   Flask: GET / y POST /api/analizar
  web/templates/, web/static/       interfaz
scripts/verificar_compatibilidad_mel.py
```

`clonvoz/caracteristicas.py` no importa Flask ni matplotlib: el procesamiento se
puede usar y probar sin levantar la aplicación.

## Etapa 2 — Perceiver e identidad de voz

La misma página incorpora un botón **Analizar etapa 2**. Usa la referencia
grabada o cargada arriba, después de ejecutar **Analizar** en la etapa 1.
La etapa 2 recibe exactamente el mel completo calculado en ese análisis.
No vuelve a enviar el WAV ni recalcula el mel. Conserva el mismo identificador.
Confirma el consentimiento antes de grabar o procesar las voces.

Para comparar: analiza la primera referencia en etapa 1, pulsa **Conservar
análisis para comparar**, carga otra referencia arriba y ejecuta su etapa 1.
Después pulsa **Analizar etapa 2**: ambas entradas serán mels ya calculados.
Al cambiar de referencia el botón de etapa 2 se bloquea hasta terminar etapa 1.

Esta etapa muestra pesos **reales** de atención y el condicionamiento de voz y
estilo `[32, 1024]`. No genera texto ni audio. No realiza identificación biométrica.
La etapa 1 sigue disponible aunque falte el checkpoint de la etapa 2.

### Entorno

Se usa el mismo entorno `.venv` de la instalación; no hace falta uno aparte.

Para GPU se necesita una instalación de PyTorch/torchaudio compatible con CUDA;
la instalación CPU no habilita CUDA por sí mismo. El código selecciona GPU
si está disponible y permite forzar CPU con `$env:CLONVOZ_DISPOSITIVO = "cpu"`.

### Pesos oficiales

Lee la [ficha y licencia del modelo](https://huggingface.co/coqui/XTTS-v2).
El proyecto y estos pesos se usan con fines académicos conforme a CPML.
La descarga ocupa aproximadamente 1.9 GB y se hace una sola vez:

```powershell
python scripts/descargar_modelo_etapa2.py --uso-academico
python app.py
```

Abre http://127.0.0.1:5000. La descarga usa la revisión oficial `v2.0.2`.
Si ya tienes el modelo, coloca `config.json` y `model.pth` en
`modelos/xtts_v2/`, o indica su carpeta con `$env:CLONVOZ_MODELO = "C:\ruta\modelo"`.
No se descarga nada al abrir la web. El modelo se carga al ejecutar la etapa 2.
Los pesos y el entorno están excluidos de Git.

### Qué significan las gráficas

1. La etapa 1 carga, valida y analiza el WAV como antes, mostrando sus cuatro
   gráficas. El servidor conserva solo su mel `[80,T]` y metadatos, sin muestras WAV.
2. El navegador recibe `id_corrida`. Al pulsar etapa 2 envía ese identificador,
   no el audio ni el tensor. El servidor recupera el mel de la misma sesión.
3. `identidad_desde_mel` entrega esa matriz a `procesar_mel`. Se divide por
   `mel_stats` del checkpoint, sin modificar la matriz original de etapa 1.
4. Los módulos oficiales `ConditioningEncoder` y `PerceiverResampler` procesan
   el mel completo y producen `[32,1024]`. La atención al audio tiene `[32,T]`.
5. Un hook observa `conditioning_perceiver.layers[-1][0].attend.attn_dropout`.
   En evaluación captura las probabilidades usadas sin cambiar el cálculo.
   Se muestra el promedio de las ocho cabezas de la última capa.

La atención completa es `[8,32,32+T]`: las primeras 32 columnas corresponden a
las propias consultas. El mapa muestra solo la parte del audio, sin renormalizar.
Por eso sus filas no tienen por qué sumar uno.

**Este prototipo utiliza el mel completo, como indica el documento de etapas.**
No reproduce el promedio por fragmentos de `get_gpt_cond_latents` del flujo de
síntesis. Su equivalencia se comprueba contra `get_style_emb` oficial sobre el
mismo mel completo. Esa diferencia es intencional y no altera la etapa 1.

Al comparar referencias, ambos mapas usan las mismas escalas de color. Las
diferencias pueden depender del hablante, contenido y estilo; no son una prueba
de identificación de personas.

El puente conserva hasta ocho mels durante diez minutos. Cada entrada pertenece
a una sesión, se libera al descartarla o reemplazarla (excepto si se conserva
para comparar), y caduca automáticamente aunque no haya nuevas peticiones.
La página intenta liberarlos al cerrarse; el temporizador cubre cierres bruscos.
Tras caducar, reiniciar el servidor o desalojar una entrada por el límite, la web
pide ejecutar de nuevo etapa 1 y no recalcula silenciosamente el audio.

### Terminal y exportación voluntaria

```powershell
python -m clonvoz.cli_identidad audios/muestra_hablante.wav --consentimiento
python -m clonvoz.cli_identidad audios/muestra_hablante.wav --consentimiento --salida salidas/etapa2
```

La CLI encadena internamente etapa 1 y etapa 2, pasando el mel sin repetirlo.
Sin `--salida` solo imprime el resumen. Con esa opción guarda `identidad.npy`,
`atencion.npy`, `atencion_consultas.npy`, `identidad.json` y los mapas en ambos
temas. El JSON contiene el intervalo completo, formas, identificador y tiempos. En la web
no se guardan audios ni resultados en disco; el servidor conserva los pesos
y el mel temporal necesario para comunicar las etapas. Los tensores completos no se envían al navegador.

### Archivos nuevos y requisitos cubiertos

| Archivo | Responsabilidad |
|---|---|
| `clonvoz/identidad.py` | Carga selectiva de pesos, hooks y procesamiento |
| `clonvoz/graficas_identidad.py` | Mapas y escalas compartidas |
| `clonvoz/web/identidad.py` | Codificación por identificador y descarte |
| `clonvoz/web/memoria.py` | Puente temporal de mels entre etapas y sesiones |
| `clonvoz/web/static/js/identidad.js` | Ejecución y comparación en la página |
| `clonvoz/cli_identidad.py` | Ejecución independiente de Flask |
| `scripts/descargar_modelo_etapa2.py` | Preparación del checkpoint |
| `scripts/verificar_etapa2.py` | Verificación con los pesos reales |
| `tests/test_identidad.py` | Reutilización del mel, sesiones, caducidad y regresión de etapa 1 |

Se conserva RF-01–04; se aplica RF-08, la parte Perceiver de RF-11 y RF-20.
Se reutiliza el diseño para RNF-04–07, RNF-09–10; se implementan CPU/GPU
(RNF-11), consentimiento (RNF-12), retención limitada (RNF-13), uso académico
(RNF-14), independencia del servidor (RNF-15) y versiones fijadas (RNF-16).
El punto de captura documentado arriba cubre la trazabilidad del módulo de
RNF-17; todavía no se emiten eventos WebSocket. El resto de la inferencia,
streaming y modo grabado se pospone a sus etapas.

### Comprobación

```powershell
python -m unittest discover -s tests -v
python scripts/verificar_etapa2.py
```

La segunda comprobación requiere pesos reales y compara la salida instrumentada
con los mismos módulos sin hooks. Usa una señal artificial explícita: comprueba
el cálculo y la captura, no la calidad perceptual con voces humanas.

La verificación comprueba que etapa 2 no recalcula el mel, conserva el
identificador y coincide con `get_style_emb` oficial sobre el mel completo.
También se comprueban caducidad, aislamiento entre sesiones, descarte,
comparación de dos señales artificiales en la web y exportación desde la CLI.
La ejecución en GPU y la comparación con grabaciones de hablantes reales quedan
pendientes de comprobar en el equipo y con las referencias correspondientes.

Código de referencia: [GPT y módulos de condicionamiento](https://github.com/idiap/coqui-ai-TTS/blob/v0.27.5/TTS/tts/layers/xtts/gpt.py),
[Perceiver y atención](https://github.com/idiap/coqui-ai-TTS/blob/v0.27.5/TTS/tts/layers/xtts/perceiver_encoder.py),
[normalización y fragmentos](https://github.com/idiap/coqui-ai-TTS/blob/v0.27.5/TTS/tts/models/xtts.py).

### Cómo leer las nuevas vistas de etapa 2

Se observan las dos capas del Perceiver sin cambiar sus operaciones. Cada mapa
promedia ocho cabezas y muestra atención al audio; el pie informa cuánto peso
reciben el audio y las propias consultas. La escala de color PowerNorm(0.4)
revela pesos pequeños sin renormalizarlos. No se impone un mínimo artificial
al máximo de la escala. Si todos los pesos al audio son cero, se muestra un
mensaje en lugar de un mapa negro. Esto no demuestra que otras capas no usaron
el audio. Las referencias comparadas comparten escala dentro de cada vista.

La vista principal del condicionamiento muestra cosenos entre las 32 consultas,
no similitud de personas ni calidad de clonación. Las consultas nulas no tienen
coseno definido. Dos detalles desplegables muestran las diferencias respecto a
la media por dimensión y la matriz original. Restar la media solo afecta esa
vista: identidad.npy y los vectores usados por las siguientes etapas permanecen
intactos. No se calcula un porcentaje de parecido entre hablantes.

scripts/verificar_compatibilidad_mel.py sigue comprobando el log-mel contra
wav_to_mel_cloning. scripts/verificar_etapa2.py comprueba además la normalización,
la igualdad con la salida oficial y la captura de ambas capas sin dejar hooks.

## Etapa 3 — GPT-2 autoregresivo: tokens de audio

La sección **Etapa 3** de la misma página toma los 32 vectores de la etapa 2 y
un texto en español (hasta 239 caracteres) y muestra en vivo cómo el GPT-2 de
XTTS-v2 genera tokens de audio de uno en uno: las fichas BPE del texto, cada
token con su probabilidad, el progreso sobre el tope de 602 y el mapa de
atención de las capas 1, 15 y 30. El resultado de esta etapa es la secuencia; la
voz la da la etapa 4. Requiere haber ejecutado la etapa 2 para la referencia
actual; la etapa 3 usa esos vectores y no los recalcula.

No se reimplementa el modelo: se llama a `GPT.generate` oficial con un
streamer, un criterio de parada (para **Detener**) y un procesador de logits
pasivo. La atención se lee con ganchos en modo `eager`, porque SDPA no devuelve
los pesos; los tokens son los mismos. La transmisión es por Server-Sent Events.

**Guardar corrida** descarga los eventos en JSON; **Reproducir corrida** los
vuelve a animar sin servidor ni modelo. Nada se guarda en el servidor.

### Interpretabilidad: qué se ve y qué no prueba

**La atención no es una explicación.** Que una posición reciba mucho peso no
prueba que haya causado la decisión: es un indicio descriptivo. En la literatura
el debate sigue abierto (Jain y Wallace, 2019, frente a Wiegreffe y Pinter,
2019) y este proyecto no lo resuelve; lo reconoce y añade una medida distinta.

| Limitación del mapa en vivo | Qué se hizo |
|---|---|
| Atención no es causa | **Intervenciones.** Sobre la secuencia ya generada se anula la atención de los tokens de audio hacia la voz o hacia el texto (en todas las capas y por tercios) y se mide cuánto cae el log p del token elegido, la divergencia KL y en qué fracción de pasos cambia el token más probable. La página lo muestra junto a la atención que recibía esa zona. También se puede apagar, una a una, las cabezas de una capa. |
| Se promedian las cabezas | Los ganchos guardan, **por cabeza**, cuánto peso va a la voz, al texto y al audio. La rejilla lo muestra sin promediar, y el mapa completo de cualquier cabeza se pide con un clic. |
| Solo 3 de 30 capas | La rejilla cubre **las 30 capas** en vivo (30 × 16 cabezas × 3 zonas, unos 2 KB por paso). El mapa completo existe para cualquier capa. Las tres pestañas siguen siendo una muestra. |
| Escala relativa | El reparto por zonas y el mapa por cabeza usan **una sola escala**. El mapa en vivo tiene un conmutador de escala absoluta (0 a 1); por omisión sigue escalando cada fila a su máximo, que revela más detalle. |
| Modo `eager` | Sigue siendo necesario. `verificar_etapa3.py` comprueba que los tokens son idénticos. |
| Observar no cambia la generación | Sigue siendo cierto: los ganchos son pasivos. Las intervenciones ocurren después y no tocan los tokens ni el audio. |

Las intervenciones usan `head_mask`, el argumento estándar de GPT-2 que
multiplica los pesos de atención después del softmax, sobre una pasada con
*teacher forcing* del mismo `gpt_inference` oficial. `verificar_etapa4.py`
comprueba que esa pasada ve la misma atención que la generación en vivo
(diferencia máxima del orden de 1e-6).

Lo que estas medidas **no** dicen: bloquear atención deja al modelo en una
situación que no vio al entrenar, así que mide dependencia, no una
"importancia" única; dos zonas redundantes pueden parecer poco necesarias por
separado; y la probabilidad que se compara es la del modelo antes de los
filtros de muestreo (temperatura, top-k, top-p y penalización por repetición).

### Escuchar por fases

Audio crudo de cada transformación, calculado solo al pedirlo: la voz original,
la voz reconstruida desde su mel (Griffin-Lim), la voz convertida en códigos del
DVAE y de vuelta, y los tokens del GPT-2 decodificados por el DVAE. Suena
metálico y apagado a propósito: es la representación intermedia, no la voz
final. La etapa 2 no tiene fase audible. Las fases 3 y 4 necesitan `dvae.pth`.

XTTS-v2 no usa el DVAE para sintetizar: entrega los estados ocultos del GPT al
HiFi-GAN. Aquí el DVAE sirve para escuchar qué significan los tokens; la etapa 4
sigue el camino real.

### Archivos y ejecución

El script de descarga ahora baja también `vocab.json` y `dvae.pth`; los que ya
existan se conservan:

```powershell
python scripts/descargar_modelo_etapa2.py --uso-academico
python app.py
python -m clonvoz.cli_generacion audios/muestra_hablante.wav --texto "Hola, ¿cómo estás?" --consentimiento
python -m clonvoz.cli_generacion audios/muestra_hablante.wav --texto "Hola" --consentimiento --salida salidas/etapa3
```

La CLI encadena las etapas 1 a 3. Con `--salida` guarda `tokens.npy`,
`corrida.json` (reproducible en la página) y los WAV de las fases 2 a 4.
`--determinista` elige siempre el token más probable; `--semilla` fija el muestreo.

En CPU cada token tarda decenas de milisegundos y el modelo tarda unos segundos
en cargarse la primera vez. El modo `eager` es algo más lento que SDPA.

| Archivo | Responsabilidad |
|---|---|
| `clonvoz/generacion.py` | GPT-2 oficial instrumentado: tokens, probabilidad y atención |
| `clonvoz/explicacion.py` | Atención de cualquier capa y cabeza, e intervenciones causales |
| `clonvoz/fases.py` | Griffin-Lim y DVAE para las fases audibles |
| `clonvoz/web/generacion.py` | `POST /api/generar` (SSE), `POST /api/explicar/atencion`, `POST /api/explicar/causal` y `POST /api/fases/audio` |
| `clonvoz/web/modelos.py` | Modelos cargados una sola vez y compartidos entre etapas |
| `clonvoz/web/static/js/generacion.js` | Vista en vivo, mapa de atención, corridas y fases |
| `clonvoz/cli_generacion.py` | Ejecución independiente de Flask |
| `scripts/verificar_etapa3.py` | Verificación con los pesos reales |
| `tests/test_generacion.py` | Generación, cancelación, SSE y fases con modelos reducidos |
| `tests/test_explicacion.py` | Reparto por cabeza, atención recalculada e intervenciones |

### Comprobación

```powershell
python -m unittest discover -s tests -v
python scripts/verificar_etapa3.py
```

La segunda requiere los pesos reales. Ejecuta `Xtts.inference` oficial hasta
el GPT para obtener sus argumentos exactos y comprueba que la tokenización y
los tokens (en modo determinista y con semilla) son idénticos a los de la
etapa 3 instrumentada. También mide los milisegundos por token en `eager` y
SDPA, y el error de ida y vuelta del DVAE.

## Etapa 4 — De los tokens a la voz: DVAE y HiFi-GAN

La sección **Etapa 4** convierte los tokens de la etapa 3 en audio reproducible
a 24 000 Hz. Es la primera etapa en la que se oye la voz clonada.

El documento de etapas describe el camino *tokens → DVAE → mel → HiFi-GAN*. El
código oficial de XTTS-v2 no hace eso, y aquí se sigue el código:

1. **Latentes.** El GPT-2 vuelve a leer la secuencia que generó
   (`gpt(texto, largo, códigos, largo_esperado, cond_latents=..., return_latent=True)`)
   y entrega sus estados ocultos: `[N, 1024]`, uno por token.
2. **Vector de hablante.** El codificador de hablante del propio HiFi-GAN resume
   la referencia, remuestreada a 16 kHz, en `[512]`. Necesita el WAV, que el
   servidor no guardó: el navegador lo reenvía una vez y se comprueba que su mel
   sea el de la referencia analizada. Después solo se conserva el vector.
3. **HiFi-GAN.** Latentes + vector de hablante → onda.

El DVAE queda como **rama didáctica**: su mel muestra qué describen los tokens.
La página enseña tres mels con la misma escala de color: la referencia (otro
texto: se compara timbre y rango, no contenido), el del DVAE y el de la voz que
salió del vocoder. Los dos últimos son la misma frase y deben parecerse aunque
el vocoder no haya leído el del DVAE; el resumen da su diferencia media.

**Escuchar sin vector de hablante** es una intervención audible: los mismos
latentes pasan otra vez por el vocoder con ese vector en cero. Lo único que
cambia es el vector, así que la diferencia que se oye es lo que aporta.

```powershell
python app.py
python -m clonvoz.cli_sintesis audios/muestra_hablante.wav --texto "Hola, ¿cómo estás?" --consentimiento --salida salidas/etapa4
```

La CLI escribe `voz_clonada.wav`, `tokens.npy`, `latentes.npy`, `corrida.json`,
`causal.json` y las figuras. `--sin-explicacion` omite las intervenciones.

El volumen del WAV se normaliza al escribirlo. El condicionamiento del GPT-2
sigue siendo el mel completo de la etapa 2, no el promedio por fragmentos de
`get_gpt_cond_latents`: con la misma identidad, el resultado es el de
`Xtts.inference`.

| Archivo | Responsabilidad |
|---|---|
| `clonvoz/sintesis.py` | Latentes, vector de hablante y HiFi-GAN oficiales |
| `clonvoz/graficas_sintesis.py` | Los tres mels con escala compartida y la onda |
| `clonvoz/web/sintesis.py` | `POST /api/sintetizar` y `POST /api/sintetizar/contraste` |
| `clonvoz/web/static/js/sintesis.js` | Reproductor, figuras y contraste |
| `clonvoz/cli_sintesis.py` | Corrida completa sin Flask |
| `scripts/verificar_etapa4.py` | Comparación con `Xtts.inference` oficial |
| `tests/test_sintesis.py` | Camino de latentes, web y contraste con modelos reducidos |

## Etapa 5 — Corrida completa en tiempo real

**Ejecutar todo**, arriba de la página, corre las cuatro etapas en una sola
pasada. El diagrama de arquitectura (ver etapa 6) se anima conforme el modelo
avanza, y cada panel de detalle se llena con los mismos datos que cuando la
etapa se ejecuta sola. Pulsar un bloque lleva a su panel.

```
POST /api/inferencia              audio + texto -> 202 {id_corrida, ws}
WS   /ws/corrida/<id>             un mensaje JSON por evento, en el momento en que ocurre
GET  /api/inferencia/<id>/audio   el WAV final
```

Eventos del canal, cada uno `{tipo, t, datos}` con `t` en milisegundos desde el
inicio: `corrida`, `bloque` (`activo`, `listo` u `omitido`, con milisegundos y
forma del tensor, para los diez bloques: audio, mel, Perceiver, tokenizador BPE,
GPT-2, tokens, DVAE, codificador de hablante, HiFi-GAN y audio clonado), `etapa1`, `etapa2`, `inicio` / `paso` / `fin` (los de la
etapa 3), `etapa4` (figuras y audio), `causal` y `terminada` o `error`. Nunca
viaja un tensor completo. Enviar `{"accion": "detener"}` por el canal, o cerrar
la página, detiene el GPT-2 en el siguiente token.

`clonvoz/pipeline.py` no reescribe nada: llama en orden a `analizar`,
`identidad_desde_mel`, `GeneradorTokens.generar`, el DVAE y `Sintetizador`, y
avisa de cada paso con una función `emitir`. No conoce Flask ni JSON; la misma
función sirve a la CLI. El modelo corre en un hilo aparte y deja los eventos en
un registro que el canal retransmite, así que la interfaz no se bloquea y quien
se conecta un instante tarde los recibe desde el primero. Solo puede haber una
corrida a la vez. Las intervenciones causales se calculan después de entregar el
audio, para no retrasar la voz.

Al terminar, la corrida queda en la memoria de la sesión como si se hubieran
ejecutado las etapas una por una: se puede pedir el mapa de otra cabeza o el
contraste sin vector de hablante. **Guardar corrida completa** descarga todos
los eventos, figuras y audio incluidos; **Reproducir corrida completa** los
vuelve a animar sin servidor ni modelo. Es la base del modo grabado de la
etapa 6.

Se añade `flask-sock` para el canal WebSocket; funciona con el servidor de
desarrollo de Flask en modo `threaded`. La generación por SSE de la etapa 3
(`/api/generar`) y las demás rutas por etapa se conservan.

| Archivo | Responsabilidad |
|---|---|
| `clonvoz/pipeline.py` | Las cuatro etapas en una corrida que emite eventos |
| `clonvoz/web/pipeline.py` | HTTP, registro de corridas y canal WebSocket |
| `clonvoz/web/static/js/pipeline.js` | Diagrama en vivo, reparto a los paneles, guardar y reproducir |
| `tests/test_pipeline.py` | Orden de eventos, cancelación y un servidor real con WebSocket |

### Comprobación de las etapas 4 y 5

```powershell
python -m unittest discover -s tests -v
python scripts/verificar_etapa4.py
```

La segunda requiere los pesos reales y unos 5 GB de memoria. Carga `Xtts` por el
camino oficial y comprueba que el vector de hablante, los latentes y la onda de
la etapa 4 son los de `Xtts.inference` (en modo determinista y con semilla), que
la atención recalculada coincide con la observada en vivo y cuánto tardan las
intervenciones. Usa una señal artificial: comprueba el cálculo, no la calidad
perceptual. En la CPU de desarrollo (6 hilos) una frase corta tarda alrededor de
un minuto de principio a fin: unos 110 ms por token en el GPT-2, el vocoder a la
mitad del tiempo real y unos 20 a 40 s las intervenciones.

Al redirigir la salida de los scripts a un archivo en Windows, define
`$env:PYTHONIOENCODING = "utf-8"`: las fichas BPE usan el carácter `␣`.

La calidad de la clonación con grabaciones de hablantes reales y la ejecución en
GPU quedan pendientes de comprobar en el equipo y con las referencias
correspondientes.

## Etapa 6 — Frontend de exposición y modo grabado

La página queda lista para presentarse: menos texto a la vista (las
explicaciones largas están en desplegables «Cómo leerlo»), modo claro y oscuro,
y transiciones de 200 a 400 ms en cada cambio de estado.

**Arquitectura en vivo.** El diagrama reproduce la arquitectura de XTTS-v2 en
cuatro grupos:

- **Contenido:** texto → tokenizador BPE.
- **Identidad:** audio de referencia → mel a 22 050 Hz → condicionamiento y
  Perceiver; y a 16 kHz → codificador de hablante.
- **Generación:** GPT-2 ⇄ tokens de audio, con la realimentación.
- **Decodificación:** HiFi-GAN → audio clonado.

El DVAE aparece punteado: aquí es una rama didáctica, y en XTTS solo se usa al
entrenar.

Cuando un bloque empieza, un punto recorre las conexiones que llegan a él, y
cada token generado da una vuelta GPT-2 → tokens → GPT-2. Cada bloque muestra
su estado, el tiempo y la forma del tensor. Con «reducir movimiento» del
sistema solo se iluminan las conexiones. Está hecho en SVG, en
`clonvoz/web/static/js/arquitectura.js`, sin librerías.

**Seguir la corrida.** Durante una corrida, en vivo o grabada, la página baja
sola a cada etapa: gráficas de la etapa 1, Perceiver, tokens del GPT-2, voz y
efecto causal. Si haces scroll a mano, el seguimiento se pausa y aparece
«Volver a seguir». Una barra fija arriba muestra los diez bloques y la etapa en
curso mientras el diagrama queda fuera de la vista.

**Modo grabado (respaldo).** El interruptor **En vivo / Grabado** cambia entre
el modelo real y una corrida guardada, y ambos animan todo igual. El modo
grabado tiene reproducir, pausa, reanudar, detener y velocidad de 0.5× a 4×,
con los tiempos originales (las esperas de más de 1 s, como cargar el modelo,
se recortan). No necesita el modelo cargado ni conexión con el servidor si se
abre un archivo.

Para preparar el respaldo antes de la exposición, con tu voz y tu consentimiento:

```powershell
python -m clonvoz.cli_sintesis audios/mi_voz.wav --texto "Hola, esta es mi voz clonada." --consentimiento --grabar grabaciones/demo.json
```

o pulsa **Guardar corrida** tras una corrida en vivo y copia el archivo a
`grabaciones/`. La página lista esa carpeta (`GET /api/grabaciones`); otra se
indica con `$env:CLONVOZ_GRABACIONES`. La carpeta está excluida de Git porque
contiene tu voz.

| Archivo | Responsabilidad |
|---|---|
| `clonvoz/grabacion.py` | Formato de corrida grabada; misma traducción de eventos para la web y la CLI |
| `clonvoz/web/static/js/arquitectura.js` | Diagrama SVG, punto de datos y estados de los bloques |
| `clonvoz/web/static/js/pipeline.js` | En vivo / grabado, seguimiento del scroll, barra fija y reproductor |

## Etapa 7 — Reconocimiento del locutor

La salida del Perceiver (los 32 vectores de 1024 que condicionan al GPT-2) sirve
también como huella del hablante. En el diagrama, el Perceiver manda una flecha
al bloque **Identificación de locutor**, que indica qué modelo decidió (KNN o
GMM) y el nombre que reconoció. Esto pasa en cada corrida completa, en vivo o
grabada. Si todavía no hay un reconocedor entrenado, el bloque queda «omitido» y
la clonación sigue igual.

### Preparar los audios

Una subcarpeta por locutor; el nombre de la subcarpeta es la etiqueta:

```
datos/locutores/
  Daniel/  frase1.wav  frase2.mp3  frase3.m4a  …
  Ana/     …
```

Se aceptan WAV, MP3, M4A, FLAC, OGG, OPUS, AAC, WMA y WEBM. WAV, FLAC y OGG se
leen igual que en la etapa 1 (libsndfile y librosa); el resto lo decodifica
**FFmpeg**, que debe estar en el PATH (o en la ruta que indique
`$env:CLONVOZ_FFMPEG`). Todo se pasa a mono y 22 050 Hz.

Esta etapa tiene **límites propios**: de 1 a 20 s por audio, configurables con
`--duracion-minima` y `--duracion-maxima`. Los de la referencia de clonación (3 a
30 s) no cambian. Los audios no se parten. Se omiten, y quedan anotados en el
reporte, los audios fuera de límites, los silenciosos, los que no se pueden leer
y los locutores que tengan menos de 2 audios válidos.

### Entrenar

```powershell
python -m clonvoz.cli_locutores --consentimiento entrenar datos/locutores
```

1. Cada audio pasa por el mel y el Perceiver. Los vectores se guardan en
   `modelos/locutores/embeddings.npz`: al reentrenar solo se procesan los
   archivos nuevos o modificados.
2. **Validación cruzada de 5 pliegues por locutor.** En cada pliegue se usan 4
   audios de cada locutor para entrenar y 1 para probar; cada audio se prueba una
   sola vez. Los dos modelos usan exactamente la misma partición.
3. **KNN:** se prueban k = 1, 3 y 5 con distancia coseno y euclidiana, y se
   queda la mejor combinación.
4. **GMM:** un modelo por locutor, con 8 componentes fijos (`--componentes`).
5. Se elige el modelo con **mayor exactitud**; si empatan, el de mayor recall
   macro, y si empatan en ambos, el KNN. La página muestra cuál se está usando y
   por qué.
6. Los dos modelos se reentrenan con todos los audios y se guardan.

Se escribe en `modelos/locutores/`, que está excluida de Git (`$env:CLONVOZ_LOCUTORES`
cambia la carpeta):

| Archivo | Contenido |
|---|---|
| `reconocedor.joblib` | Los dos modelos entrenados, sus umbrales y cuál está en uso |
| `reporte.json` | Todos los resultados del entrenamiento |
| `reporte/*.png` | Matrices de confusión de KNN y GMM y proyección PCA |
| `reporte/*.csv` | Métricas, búsqueda del KNN, recall por locutor, matrices, predicción de cada audio y audios omitidos |

La página lee el modelo y el reporte al abrirse; si reentrenas con el servidor
abierto, pulsa **Recargar reporte**. Desde la página también se descargan los
archivos uno a uno o todos juntos en un `.zip`.

Para probar un audio suelto: `python -m clonvoz.cli_locutores --consentimiento identificar audio.m4a`.

`reconocedor.joblib` es un pickle: carga solo archivos que hayas generado tú.

### Entradas y normalizaciones

Cada normalización se ajusta solo con los audios de entrenamiento de cada
pliegue.

**KNN** (un vector por audio)
1. Se promedian las 32 consultas → un vector de 1024.
2. Estandarización z por dimensión. Las dimensiones del Perceiver tienen escalas
   muy distintas, y sin esto unas pocas dominarían la distancia euclidiana.
3. Votos ponderados por 1/distancia.

**GMM** (las 32 consultas son 32 observaciones del locutor)
1. Centrado por consulta: a cada consulta se le resta su media en el
   entrenamiento. Cada consulta tiene una dirección común a todas las personas, y
   al quitarla queda lo que distingue a cada locutor (la misma idea que la
   normalización por media cepstral de los MFCC).
2. Estandarización z por dimensión.
3. PCA a 64 dimensiones (`--dimensiones`). Con 4 audios hay 128 observaciones
   por locutor, que no alcanzan para estimar gaussianas en 1024 dimensiones.
4. Covarianza diagonal y regularización 1e-3, por la misma razón.
5. Gana el locutor cuyo GMM da la mayor log-verosimilitud media a las 32
   observaciones.

**Locutor desconocido.** Con las predicciones de prueba de la validación
cruzada se fija el umbral que acepta al 95 % de los aciertos (`--aceptacion`).
Para el KNN es la distancia al audio más cercano del locutor; para el GMM, la
log-verosimilitud media. Un audio que no pasa el umbral se reporta como
«Desconocido», junto con el locutor más parecido. El reporte indica qué parte de
los errores habría atrapado ese umbral.

**Métricas.** *Exactitud*: aciertos entre todos los audios de prueba. *Recall
(macro)*: la fracción reconocida de cada locutor, promediada entre locutores.
Con el mismo número de audios por locutor ambas coinciden. Hay una advertencia:
la k y la distancia del KNN se eligen con los mismos pliegues con los que se
reportan, así que su exactitud puede ser algo optimista.

| Archivo | Responsabilidad |
|---|---|
| `clonvoz/locutores.py` | Lectura de audios (FFmpeg), extracción, KNN, GMM, validación cruzada, umbrales, guardado |
| `clonvoz/graficas_locutores.py` | Matrices de confusión, proyección PCA y exportación PNG/CSV |
| `clonvoz/cli_locutores.py` | `entrenar` e `identificar` desde la terminal |
| `clonvoz/web/locutores.py` | `/api/locutores`, exportación y `/api/locutor` |
| `clonvoz/web/static/js/locutores.js` | Sección «Etapa 7» de la página |
| `tests/test_locutores.py` | Pruebas con vectores sintéticos, MP3/M4A con FFmpeg y rutas web |
