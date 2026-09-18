## Requisitos del Sistema

- **Sistema Operativo:** Windows 10/11 (probado en Windows 64-bit).
- **Versión de Python:** **Python 3.11.x** (probado con Python 3.11.9).
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

Compara el mel-espectrograma contra el `TorchMelSpectrogram` real de XTTS-v2 y
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