# Clonador de Voz — Etapa 1 (DSP) y Etapa 2 (Perceiver / XTTS-v2)

Sistema de análisis acústico y extracción de identidad de voz basado en la arquitectura de **XTTS-v2**. El proyecto procesa señales de audio (WAV), genera representaciones en tiempo-frecuencia (forma de onda, STFT, Mel-espectrograma de 80 canales y contorno de entonación F0), y calcula el condicionamiento latente de hablante (`[32, 1024]`) junto con mapas de atención reales mediante el módulo **Perceiver Resampler**.

---

## Inicio Rápido (Cómo ejecutar el código)

Sigue estos sencillos pasos para levantar el proyecto desde cero:

### 1. Requisitos Previos del Sistema
- **Sistema Operativo:** Windows 10/11 (64-bit).
- **Versión de Python Requerida:** **Python 3.11.9** (soportado Python 3.11.x o 3.10.x de 64-bit).

### 2. Crear el Entorno Virtual
Abre una terminal (PowerShell o CMD) en la carpeta raíz del proyecto y crea el entorno con la versión específica de Python:

```powershell
# Usando el lanzador de Python en Windows (asegura usar la versión 3.11):
py -3.11 -m venv .venv

# O directamente si python ya apunta a Python 3.11:
python -m venv .venv
```

### 3. Activar el Entorno Virtual
- **En PowerShell:**
  ```powershell
  # (Si PowerShell restringe scripts en tu equipo, ejecuta primero una vez:
  #  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
  .\.venv\Scripts\Activate.ps1
  ```
- **En CMD (Símbolo del sistema):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

*(Sabrás que está activado porque tu terminal mostrará el prefijo `(.venv)`).*

### 4. Instalar las Dependencias
Actualiza el gestor `pip` e instala las librerías verificadas desde `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Descargar los Pesos del Modelo (Solo la primera vez)
Para utilizar la **Etapa 2 (Perceiver / Identidad de voz)** se requiere el checkpoint oficial de XTTS-v2 (~1.9 GB en `modelos/xtts_v2/`). Se descarga con fines académicos:

```powershell
python scripts/descargar_modelo_etapa2.py --uso-academico
```
*(Si los archivos `model.pth` y `config.json` ya están presentes en la carpeta `modelos/xtts_v2/`, el script lo detectará y los conservará).*

### 6. Ejecutar la Aplicación Web
Inicia el servidor Flask local:

```powershell
python app.py
```

Abre en tu navegador (Google Chrome o Microsoft Edge):
**http://127.0.0.1:5000**

> [!IMPORTANT]
> **Usa siempre `127.0.0.1`, no la IP local de red (ej. 192.168.x.x).** Los navegadores modernos solo habilitan el permiso del micrófono en contextos seguros (HTTPS o `localhost`/`127.0.0.1`).

---

## Scripts del Proyecto y Orden de Ejecución

A continuación se detalla la secuencia recomendada para configurar, validar y utilizar el sistema:

### Diagrama de Flujo y Orden Cronológico

```
[1. Entorno]
 py -3.11 -m venv .venv  -->  pip install -r requirements.txt
        │
        ▼
[2. Pesos del Modelo (Una sola vez)]
 python scripts/descargar_modelo_etapa2.py --uso-academico
        │
        ▼
[3. Verificaciones y Pruebas (Opcional)]
 ├── python scripts/verificar_compatibilidad_mel.py audios/Dany1.wav
 ├── python scripts/verificar_etapa2.py
 └── python -m unittest discover -s tests -v
        │
        ▼
[4. Ejecución Principal]
 python app.py  ===>  Abrir http://127.0.0.1:5000 (UI Web Completa)
        │
        ▼
[5. Flujos Alternativos por Consola (CLI)]
 ├── python -m clonvoz.cli audios/Dany1.wav --salida salidas/ --guardar-mel
 └── python -m clonvoz.cli_identidad audios/Dany1.wav --consentimiento --salida salidas/etapa2
```

### Resumen de Scripts

| Orden | Script / Comando | Propósito | ¿Cuándo ejecutarlo? |
|---|---|---|---|
| **Paso 1** | `scripts/descargar_modelo_etapa2.py --uso-academico` | Descarga `config.json` y `model.pth` oficiales de Hugging Face a `modelos/xtts_v2/`. | **Obligatorio una vez** antes de usar la Etapa 2. |
| **Paso 2** | `scripts/verificar_compatibilidad_mel.py <audio.wav>` | Valida que el Mel calculado por la Etapa 1 coincida numéricamente (tolerancia `< 1e-4`) con `wav_to_mel_cloning` de XTTS-v2. | Opcional, para verificar coherencia matemática de la Etapa 1. |
| **Paso 3** | `scripts/verificar_etapa2.py` | Evalúa el Perceiver con los pesos reales, comprueba hooks sin fugas y compara la salida con `get_style_emb`. | Opcional, para validar la integración de la Etapa 2. |
| **Paso 4** | `python -m unittest discover -s tests -v` | Ejecuta la batería de 9 tests unitarios (memoria temporal, caducidad, gráficos, aislamiento de sesión). | Opcional, para asegurar que no haya regresiones de código. |
| **Paso 5** | **`app.py`** | **Inicia el servidor web Flask (Etapas 1 y 2 integradas)** con interfaz gráfica interactiva. | **Ejecución principal del proyecto**. |
| **Paso 6 (CLI 1)** | `python -m clonvoz.cli <audio.wav> --salida salidas/ --guardar-mel` | Procesa un archivo de audio por línea de comandos sin servidor web. Genera imágenes PNG y el tensor `mel.npy`. | Cuando se requiera analizar audios por lotes o sin navegador. |
| **Paso 7 (CLI 2)** | `python -m clonvoz.cli_identidad <audio.wav> --consentimiento --salida salidas/etapa2` | Encadena Etapas 1 y 2 por consola, exportando `identidad.npy`, `atencion.npy` y `identidad.json`. | Para análisis de identidad y atención por terminal. |
| **Paso 8 (Extra)** | `python main.py` | Demostración mínima de síntesis de voz con XTTS-v2 (`tts_to_file`). | Opcional, para pruebas de síntesis TTS de referencia. |

---

## Revisión de Librerías y Dependencias (`requirements.txt`)

El archivo [requirements.txt](requirements.txt) contiene el entorno unificado y probado en **Windows con Python 3.11.9**.

### Librerías Clave Utilizadas y su Rol:

| Librería | Versión Instalada | Rol en el Proyecto |
|---|---|---|
| **`torch`** | `2.14.0` | Motor de cómputo tensorial y ejecución del Perceiver Resampler. |
| **`torchaudio`** | `2.11.0` | Operaciones de procesamiento de señales de audio con tensores. |
| **`torchcodec`** | `0.16.0` | Decodificador multimedia requerido internamente por Coqui TTS en PyTorch ≥ 2.9. |
| **`coqui-tts`** | `0.27.5` | Implementación de XTTS-v2 (`ConditioningEncoder`, `PerceiverResampler`). |
| **`Flask`** | `3.1.3` | Servidor web HTTP, API REST (`/api/analizar`, `/api/identidad`, `/api/descartar`). |
| **`Werkzeug`** | `3.1.8` | Gestión de peticiones WSGI y control de límites de carga de archivos. |
| **`librosa`** | `0.11.0` | Algoritmos DSP: STFT, banco de filtros Mel (HTK/Slaney) y estimación de F0 (`pyin`/`yin`). |
| **`soundfile`** | `0.14.0` | Lectura y validación de archivos WAV PCM (16/24/32 bits, float). |
| **`matplotlib`** | `3.11.2` | Renderizado científico de ondas, espectrogramas y mapas de calor de atención a PNG en memoria. |
| **`numpy`** | `1.26.4` | Cálculo matricial numérico de alta precisión. |
| **`transformers`** | `4.57.6` | Componentes auxiliares y tokenizadores requeridos por Coqui TTS. |

> [!NOTE]
> **Sobre `requirements-etapa2.txt`:** Se mantiene como un perfil alternativo ligero para entornos de solo CPU que utilicen PyTorch 2.8.0 sin necesidad de `torchcodec`. Para el uso normal y completo del proyecto, **`requirements.txt` es el archivo oficial**.

---

## Etapa 1 — Preprocesamiento de audio y características visuales

La primera etapa carga o graba una referencia de voz y extrae cuatro representaciones visuales y numéricas **mediante DSP clásico puro (sin redes neuronales ni GPU)**:

1. **Forma de onda:** Amplitud en función del tiempo.
2. **Espectrograma STFT:** Densidad espectral de frecuencia (0 a 11025 Hz) calculada mediante Transformada de Fourier de Tiempo Corto.
3. **Mel-espectrograma de 80 canales:** Extraído con los parámetros exactos de XTTS-v2 para garantizar compatibilidad con la siguiente etapa:
   - Frecuencia de muestreo: **22050 Hz** (resampleo automático si el archivo viene en otra frecuencia).
   - `n_fft = 2048`, `hop_length = 256`, `win_length = 1024`.
   - Escala Mel: `htk`, normalización `slaney`, rango `0 - 8000 Hz`.
4. **Contorno de F0 (Tono Fundamental):** Estimación de la entonación vocal a lo largo del tiempo utilizando el algoritmo probabilístico **pYIN** (o YIN alternativo).

### Ejecución por Línea de Comandos (CLI)
```powershell
python -m clonvoz.cli audios/Dany1.wav --salida salidas/etapa1 --guardar-mel
```
Genera los cuatro archivos PNG (`forma_de_onda.png`, `espectrograma.png`, `mel.png`, `f0.png`), el reporte `caracteristicas.json` y el tensor `mel.npy` de dimensiones `[80, T]`.

---

## Etapa 2 — Perceiver e Identidad de Voz

Esta etapa consume directamente el Mel-espectrograma calculado en la Etapa 1 y lo procesa mediante los módulos preentrenados oficiales de **XTTS-v2**:

1. **ConditioningEncoder:** Proyecta los 80 canales del Mel a representaciones de dimensión 1024 mediante capas convolucionales y atención multi-cabeza.
2. **Perceiver Resampler:** Transforma secuencias de longitud variable en un conjunto fijo de **32 vectores latentes de dimensión 1024** (`[32, 1024]`).
3. **Captura no invasiva de Atención:** Mediante un forward hook sobre la última capa (`conditioning_perceiver.layers[-1][0].attend.attn_dropout`), se capturan las matrices reales de atención (`[32, T]`) sin alterar los cálculos originales.
4. **Matriz de Cosenos de Consultas:** Permite inspeccionar la similitud y ortogonalidad entre los 32 vectores de identidad.
5. **Comparación de Referencias:** Permite mantener un hablante A de referencia, procesar un hablante B y comparar visualmente sus mapas de atención con escalas de color compartidas.

### Privacidad y Gestión Efímera en Memoria
- El servidor **no almacena audios grabados ni tensores en disco**.
- Los Mel-espectrogramas se conservan en memoria RAM mediante un puente efímero con capacidad máxima de 8 corridas y tiempo de vida (TTL) de **10 minutos**.
- Al descartar una muestra o cerrar la pestaña, la memoria se libera de inmediato.

### Ejecución por Línea de Comandos (CLI Identidad)
```powershell
python -m clonvoz.cli_identidad audios/Dany1.wav --consentimiento --salida salidas/etapa2
```
Exporta:
- `identidad.npy` — Vector de condicionamiento latente `[32, 1024]`.
- `atencion.npy` — Matriz de atención audio-consultas `[32, T]`.
- `atencion_consultas.npy` — Atención interna entre consultas `[32, 32]`.
- `identidad.json` — Métricas numéricas, tiempos en milisegundos y distribución de masa de atención.
- Gráficas PNG en temas claro y oscuro.

---

## Estructura del Proyecto

```
ClonacionVoz/
├── app.py                              # Lanzador principal del servidor web Flask
├── main.py                             # Demostración básica de síntesis TTS
├── requirements.txt                    # Dependencias completas y unificadas (Python 3.11)
├── requirements-etapa2.txt             # Dependencia alternativa ligera para CPU
├── clonvoz/
│   ├── config.py                       # Constantes y parámetros acústicos unificados
│   ├── audio.py                        # Carga, validación y remuestreo de WAV
│   ├── caracteristicas.py              # Extracción DSP (Waveform, STFT, Mel, F0)
│   ├── graficas.py                     # Renderizado de figuras de Etapa 1
│   ├── analisis.py                     # Orquestador del análisis de audio
│   ├── identidad.py                    # Inferencia de XTTS-v2 Perceiver y hooks
│   ├── graficas_identidad.py           # Renderizado de mapas de atención y cosenos
│   ├── cli.py                          # Línea de comandos para Etapa 1
│   ├── cli_identidad.py                # Línea de comandos para Etapa 2
│   └── web/
│       ├── servidor.py                 # Rutas y controladores de Flask
│       ├── memoria.py                  # Gestor temporal de mels en RAM (TTL)
│       ├── identidad.py                # Controlador web de Etapa 2
│       ├── templates/index.html        # Plantilla HTML de la aplicación
│       └── static/                     # CSS (tokens y temas) y JavaScript
├── scripts/
│   ├── descargar_modelo_etapa2.py      # Descarga oficial de Hugging Face (~1.9 GB)
│   ├── verificar_compatibilidad_mel.py # Verificación matemática Mel vs XTTS-v2
│   └── verificar_etapa2.py             # Verificación del Perceiver con pesos reales
├── tests/
│   ├── test_graficas_identidad.py      # Tests de visualizaciones y matrices
│   └── test_identidad.py               # Tests de memoria, sesiones y regresión
├── audios/                             # Muestras de audio de prueba (.wav)
└── modelos/xtts_v2/                    # Ubicación del checkpoint (config.json y model.pth)
```

---

## Licencia y Uso Académico

Los pesos preentrenados de XTTS-v2 se distribuyen bajo la licencia **Coqui Public Model License (CPML)** y su uso en este proyecto está destinado exclusivamente a **fines académicos y de investigación**. Revisa la [ficha del modelo en Hugging Face](https://huggingface.co/coqui/XTTS-v2) para más detalles.
