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