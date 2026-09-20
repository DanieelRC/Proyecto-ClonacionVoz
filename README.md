## Requisitos del Sistema

- **Sistema Operativo:** Windows 10/11 (probado en Windows 64-bit).
- **Versión de Python:** **Python 3.10 o 3.11**. La etapa 1 se probó con
  Python 3.11.9; el entorno de la etapa 2, con Python 3.10.9.
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

Para instalar todas las versiones probadas y compatibles:

```powershell
pip install -r requirements.txt
```

La etapa 1 no necesita ninguna dependencia adicional: `librosa`, `matplotlib`,
`soundfile`, `torchaudio` y `Flask` ya están fijados en `requirements.txt`.

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

### Entorno separado

Para mantener intacto el entorno anterior, se proporciona
`requirements-etapa2.txt`. No mezcles ambos archivos de dependencias en un mismo
entorno. Estos comandos instalan la variante CPU de PyTorch:

El entorno separado fue verificado en esta computadora con Python 3.10.9 y
PyTorch 2.8.0 CPU. La instalación anterior de la etapa 1 no se reemplaza.

PyTorch se queda en 2.8.0 por una razón concreta: desde la versión 2.9,
coqui-tts exige además el paquete `torchcodec` y aborta la importación si no
está (`TTS/__init__.py`). Quedarse en 2.8.0 evita esa dependencia. Como este es
el entorno con el que se ejecuta la aplicación completa, es el que manda;
`requirements.txt` queda como registro del entorno con el que se construyó la
etapa 1.

```powershell
python -m venv .venv-etapa2
.\.venv-etapa2\Scripts\python.exe -m pip install --upgrade pip
.\.venv-etapa2\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-etapa2\Scripts\python.exe -m pip install -r requirements-etapa2.txt
```

Para GPU se necesita una instalación de PyTorch/torchaudio compatible con CUDA;
el entorno CPU anterior no habilita CUDA por sí mismo. El código selecciona GPU
si está disponible y permite forzar CPU con `$env:CLONVOZ_DISPOSITIVO = "cpu"`.

### Pesos oficiales

Lee la [ficha y licencia del modelo](https://huggingface.co/coqui/XTTS-v2).
El proyecto y estos pesos se usan con fines académicos conforme a CPML.
La descarga ocupa aproximadamente 1.9 GB y se hace una sola vez:

```powershell
.\.venv-etapa2\Scripts\python.exe scripts/descargar_modelo_etapa2.py --uso-academico
.\.venv-etapa2\Scripts\python.exe app.py
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
.\.venv-etapa2\Scripts\python.exe -m clonvoz.cli_identidad audios/muestra_hablante.wav --consentimiento
.\.venv-etapa2\Scripts\python.exe -m clonvoz.cli_identidad audios/muestra_hablante.wav --consentimiento --salida salidas/etapa2
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
.\.venv-etapa2\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-etapa2\Scripts\python.exe scripts/verificar_etapa2.py
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
