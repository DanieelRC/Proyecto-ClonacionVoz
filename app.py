"""
Punto de entrada de la aplicación web.

    python app.py

Luego abre http://127.0.0.1:5000 en Chrome o Edge.

Sobre la dirección: el micrófono del navegador solo funciona en un "contexto
seguro", es decir con HTTPS o en localhost. 127.0.0.1 cuenta como localhost, así
que ahí funciona sin certificados. Si abres la aplicación por la IP de red de la
máquina (por ejemplo 192.168.x.x), el navegador bloqueará el micrófono.
"""

from clonvoz.web.servidor import crear_app

# 127.0.0.1 es solo esta computadora: el servidor no queda expuesto a la red local.
HOST = "127.0.0.1"
PUERTO = 5000


def main() -> None:
    app = crear_app()
    print("\n  Etapa 1 — preprocesamiento de audio y características visuales")
    print(f"  Abre http://{HOST}:{PUERTO} en Chrome o Edge")
    print("  Ctrl+C para detener\n")
    # debug=False evita mostrar el traceback en el navegador ante un fallo.
    # threaded=True atiende varias peticiones a la vez, que es la razón de que las
    # gráficas se dibujen sin `pyplot` (ver el encabezado de clonvoz/graficas.py).
    app.run(host=HOST, port=PUERTO, debug=False, threaded=True)


if __name__ == "__main__":
    main()
