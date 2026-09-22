"""Punto de entrada de la aplicación web del proyecto."""

from clonvoz.web.servidor import crear_app

HOST = "127.0.0.1"
PUERTO = 5000


def main() -> None:
    app = crear_app()
    print("\n  Clonador de voz — etapa 1 (audio) y etapa 2 (Perceiver)")
    print(f"  Abre http://{HOST}:{PUERTO} en Chrome o Edge")
    print("  Ctrl+C para detener\n")
    app.run(host=HOST, port=PUERTO, debug=False, threaded=True)


if __name__ == "__main__":
    main()
