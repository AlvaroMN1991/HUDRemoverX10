import gradio as gr
import os
import numpy as np
import cv2
from typing import Optional, List
from PIL import Image
from app.lang.es import text_gradio_ui, text_general  # Importamos las cadenas de texto
from app.utils.tools import obtener_propiedades_imagen, Image
from app.utils.config import FORMATOS_TEXTO,FORMATOS_COMPATIBLES 
from app.sam.sam_loader import cargar_sam_online, segmentar_automaticamente, MascaraSegmentada

# Variables globales para mantener el estado de la segmentación
mascaras_memoria: List[MascaraSegmentada] = []  # Lista de objetos de máscaras segmentadas
imagen_base_np: Optional[np.ndarray] = None  # Imagen base en NumPy


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


# Función principal que lanza la segmentación con SAM
# Devuelve miniaturas coloreadas, la imagen combinada y las opciones del checkbox
def lanzar_segmentacion(filepath: str, modelo_clave: str):
    global mascaras_memoria, colores_memoria, imagen_base_np

    if filepath is None or not os.path.exists(filepath):
        return [], None, []

    try:
        imagen = Image.open(filepath)   
        imagen_base_np = np.array(imagen.convert("RGB"))
    except Exception as e:
        return None, None, []
    
    try:
        modelo = cargar_sam_online(modelo_clave)

        # Recuperamos las máscaras como objetos MascaraSegmentada + imagen combinada
        mascaras_memoria, imagen_combinada = segmentar_automaticamente(imagen, modelo)

        # Miniaturas coloreadas
        miniaturas_coloreadas = [m.miniatura for m in mascaras_memoria]
        opciones_selector = [str(i) for i in range(len(mascaras_memoria))]

        return miniaturas_coloreadas, imagen_combinada, gr.update(choices=opciones_selector, value=[])
    except Exception as e:
        print(f"❌ Error en segmentación: {e}")
        return [], None, []
    

# Combina las máscaras seleccionadas con la imagen base
def generar_imagen_con_mascaras_combinadas(imagen_np: np.ndarray, mascaras: list[np.ndarray], colores: list[tuple[int, int, int]]) -> Image.Image:
    overlay = imagen_np.copy()

    for mask, color in zip(mascaras, colores):
        mask = mask.astype(np.uint8)
        rgb_mask = np.zeros_like(imagen_np)
        for c in range(3):
            rgb_mask[:, :, c] = mask * color[c]
        overlay = cv2.addWeighted(overlay, 1.0, rgb_mask, 0.5, 0)

    return Image.fromarray(overlay)

# Actualiza la imagen combinada al seleccionar máscaras desde el checkbox
def actualizar_mascaras_selector(indices: List[str]) -> Image.Image:
    if not indices or imagen_base_np is None:
        return Image.new("RGB", (512, 512), (0, 0, 0))

    seleccionados = list(map(int, indices))
    mascaras = [mascaras_memoria[i].binaria for i in seleccionados]
    colores = [mascaras_memoria[i].color for i in seleccionados]

    return generar_imagen_con_mascaras_combinadas(imagen_base_np, mascaras, colores)



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
                        mascaras = gr.Gallery(label="🧩 Segmentos individuales", columns=4, rows=2, show_label=True)
                        segment_selector = gr.CheckboxGroup(label="🎯 Selecciona máscaras a mostrar", choices=[], interactive=True)
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
        segment_button.click(fn=lanzar_segmentacion, inputs=[image_input, sam_selector], outputs=[mascaras, combined_mask_preview, segment_selector])
        segment_selector.change(fn=actualizar_mascaras_selector, inputs=segment_selector, outputs=combined_mask_preview)


    #descomentar para produccion
    #page.launch()
    #debug
    page.launch(show_error=True, debug=True)
