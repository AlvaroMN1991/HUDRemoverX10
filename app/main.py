# Punto de entrada de la aplicación. Lanza la interfaz Gradio definida en interface/gradio_ui.py

from app.interface.gradio_ui import GradioInterface
from app.globals.estado_global import EstadoGlobal
from app.inpainting.InpaintingBase import _autoimport_inpainters


if __name__ == "__main__":
    estado = EstadoGlobal()
    _autoimport_inpainters()  # Llamada automática
    ui = GradioInterface(estado)
    ui.build().launch(debug=True, show_error=True)