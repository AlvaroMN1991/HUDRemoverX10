import gradio as gr
import os
from PIL import Image
from app.lang.es import text_gradio_ui, text_general  # Importamos las cadenas de texto
from app.utils.tools import obtener_propiedades_imagen, Image
from app.utils.config import FORMATOS_TEXTO,FORMATOS_COMPATIBLES 
from app.sam.sam_loader import cargar_sam_online, segmentar_automaticamente

# Esta función toma una imagen cargada por el usuario y devuelve sus propiedades básicas
def get_image_properties(filepath: str) -> str:
    if filepath is None or not os.path.exists(filepath):
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


#def lanzar_segmentacion(imagen: Image.Image, modelo_clave: str):
def lanzar_segmentacion(filepath: str, modelo_clave: str):
    if filepath is None or not os.path.exists(filepath):
        return [], None

    try:
        imagen = Image.open(filepath)   
    except Exception as e:
        return None, None
    
    try:    
        model = cargar_sam_online(modelo_clave)
        imagenes_mascaras, combined_mask = segmentar_automaticamente(imagen, model)

        return imagenes_mascaras, combined_mask
    except Exception as e:
        return None, None


# Esta función crea la interfaz gráfica con Gradio
def launch_interface():
    with gr.Blocks() as page:
        gr.Markdown(f"## {text_gradio_ui['titulo_app']}")

        with gr.Row(): 
                with gr.Column():
                # Cuadro para subida de imagen
                    with gr.Tab("Imágen original"):
                        image_input = gr.Image(type="filepath", label=f"{text_gradio_ui['subir_imagen']} {FORMATOS_TEXTO}") 
                with gr.Column():
                    with gr.Tab("Mascaras"):
                        mascaras = gr.Gallery(label="🧩 Segmentos individuales", columns=4, rows=2)
                    with gr.Tab("Mascaras Combinadas"):
                        combined_mask_preview = gr.Image(label="🧵 Vista general de todas las segmentaciones", interactive=False)
                    with gr.Tab("Imagen Final"):
                        image_output = gr.Image(label="✅ Imagen sin HUD", interactive=False)               
        
        with gr.Row(): 
            with gr.Column():
                # Cuadro con las propiedades de la imagen
                with gr.Tab(text_gradio_ui["propiedades_imagen"]):  
                    image_info = gr.Textbox(label="", lines=4, interactive=False) 
            with gr.Tab("Inpainting y Remover"):
                with gr.Column():
                    opciones_sam = [("SAM B (rápido, poca precisión)", "vit_b"), ("SAM L (Equilibrado)", "vit_l"), ("SAM H (El más preciso)", "vit_h")]
                    sam_selector = gr.Radio(label="🧠 Modelo de SAM", choices=opciones_sam,  value="vit_b" ,interactive=True)
                with gr.Column():
                    opciones_inpainting = [("OpenCV", "1"), ("LaMa", "2"), ("Stable Diffusion", "3")]
                    inpaint_selector = gr.Radio(label="🧠 Modelo de Inpainting", choices=opciones_inpainting, value="1", interactive=True)
            with gr.Tab("Editor de imágen"):                
                with gr.Column():
                    opciones_inpainting = [("OpenCV", "1"), ("LaMa", "2"), ("Stable Diffusion", "3")]
                    inpaint_selector = gr.Radio(label="🧠 Modelo de Inpainting", choices=opciones_inpainting, value="1", interactive=True)
                with gr.Column():
                    opciones_sam2 = [("SAM B (rápido, poca precisión)", "vit_b"), ("SAM L (Equilibrado)", "vit_l"), ("SAM H (El más preciso)", "vit_h")]
                    sam_selector2 = gr.Radio(label="🧠 Modelo de SAM", choices=opciones_sam2, interactive=True)
        with gr.Row():
            segment_button = gr.Button("📐 Detectar Segmentos")
            apply_button = gr.Button("🧹 Eliminar HUD")  

        # Cada vez que se suba una imagen, se llamará a esta función para analizarla
        image_input.change(fn=get_image_properties, inputs=image_input, outputs=image_info)
        segment_button.click(fn=lanzar_segmentacion, inputs=[image_input, sam_selector], outputs=[mascaras, combined_mask_preview])


    #descomentar para produccion
    #page.launch()
    #debug
    page.launch(show_error=True, debug=True)
