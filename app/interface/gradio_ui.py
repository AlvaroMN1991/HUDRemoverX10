import gradio as gr
import numpy as np
import os
from app.lang.es import text_gradio_ui, text_general  # Importamos las cadenas de texto
from app.utils.image_info import obtener_propiedades_imagen, Image
from app.config import FORMATOS_TEXTO,FORMATOS_COMPATIBLES

# Esta función toma una imagen cargada por el usuario y devuelve sus propiedades básicas
def get_image_properties(filepath: str) -> str:
    if not filepath:
        return text_gradio_ui["sin_imagen"]
  
    # Validar extensión del archivo
    extension = os.path.splitext(filepath)[1].lower()
    if extension not in FORMATOS_COMPATIBLES:
        return text_general["formato_no_soportado"].format(formatos=FORMATOS_TEXTO)
    try:
        image = Image.open(filepath)
        return obtener_propiedades_imagen(image, filepath)
    except Exception as e:
        return text_gradio_ui["error_cargar_imagen"].format(error=e)


# Esta función crea la interfaz gráfica con Gradio
def launch_interface():
    with gr.Blocks() as page:
        gr.Markdown(f"## {text_gradio_ui['titulo_app']}")

        with gr.Row(scale=0):  # No se estira más de lo necesario)
            # Cuado para subida de imagen
            image_input = gr.Image(type="filepath", label=f"{text_gradio_ui['subir_imagen']} {FORMATOS_TEXTO}", height=512, width=512) 

            # Cuadro con las propiedades de la imagen
            image_info = gr.Textbox(label=text_gradio_ui["propiedades_imagen"], lines=4) 

        # Cada vez que se suba una imagen, se llamará a esta función para analizarla
        image_input.change(fn=get_image_properties, inputs=image_input, outputs=image_info)

    page.launch()
