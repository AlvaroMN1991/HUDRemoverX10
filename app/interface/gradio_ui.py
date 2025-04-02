import gradio as gr
import os
import numpy as np
import cv2
from typing import Optional, List, Tuple
from PIL import Image
from app.lang.es import text_gradio_ui, text_general  # Importamos las cadenas de texto
from app.utils.tools import obtener_propiedades_imagen
from app.utils.config import FORMATOS_TEXTO,FORMATOS_COMPATIBLES, TipoInpainting, TipoSegmentacion
from app.sam.sam_loader import cargar_sam_online, segmentar_automaticamente, MascaraSegmentada, mejorar_imagen_para_segmentacion
from app.inpainting.InpaintingBase import InpaintingBase  
from app.inpainting.OpenCVInpainting import OpenCVInpainting #Esto lo hacemmos para que se registre la clase y podamos usarla dinamicamente
from app.inpainting.LaMaInpainting import LaMaInpainting #Esto lo hacemmos para que se registre la clase y podamos usarla dinamicamente
from app.sam.SegmentadorPuntos import SegmentadorPorPunto #Esto lo hacemmos para que se registre la clase y podamos usarla dinamicamente
from app.sam.SegmentadorBase import SegmentadorBase, Segmentacion #Esto lo hacemmos para que se registre la clase y podamos usarla dinamicamente

# Variables globales para mantener el estado de la segmentación
mascaras_memoria: List[MascaraSegmentada] = []  # Lista de objetos de máscaras segmentadas
imagen_base_np: Optional[np.ndarray] = None  # Imagen base en NumPy
puntos_usuario: List[Tuple[int, int]] = []


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

# Capturar clics del usuario
def registrar_puntos(evt: gr.SelectData):
    global puntos_usuario
    try:        
        puntos_usuario.append((evt.index[0], evt.index[1]))
        return puntos_usuario
    except Exception as e:
        return puntos_usuario
    
# Función principal que lanza la segmentación con SAM
# Devuelve miniaturas coloreadas, la imagen combinada y las opciones del checkbox
def lanzar_segmentacion(filepath: str, modelo_clave: str, metodo_segmentacion:int=0):
    global mascaras_memoria, colores_memoria, imagen_base_np

    if filepath is None or not os.path.exists(filepath):
        return [], None, []

    try:
        imagen = Image.open(filepath)   
        imagen_base_np = np.array(imagen.convert("RGB"))
    except Exception as e:
        return None, None, []
    
    try:
        modelo = cargar_sam_online(modelo_clave, modo=metodo_segmentacion)
        #modelo = cargar_sam_online(modelo_clave)

        #lo dejo comentado de momento, pero ayuda en la segmentacion.
        #imagen = Image.fromarray(mejorar_imagen_para_segmentacion(np.array(imagen.convert("RGB"))))

        if metodo_segmentacion == 0:
            # Método automático actual (temporalmente mantenido)
            # Recuperamos las máscaras como objetos MascaraSegmentada + imagen combinada
            mascaras_memoria, imagen_combinada = segmentar_automaticamente(imagen, modelo)
        else:

            # 🔥 Creamos anotación siempre (para automático serán vacíos los puntos)
            etiquetas_usuario = [1] * len(puntos_usuario) if metodo_segmentacion == 1 else []
            anotacion = Segmentacion(tipo=TipoSegmentacion(metodo_segmentacion), puntos=puntos_usuario, etiquetas=etiquetas_usuario)

            # 🔥 Creamos el segmentador usando la Factory
            segmentador = SegmentadorBase.crear(TipoSegmentacion(metodo_segmentacion), modelo)

            mascaras_memoria = segmentador.segmentar(imagen_base_np, anotacion)
  
            #colores = [m.color for m in mascaras_memoria]
            imagen_combinada = generar_imagen_con_mascaras_combinadas_mascarasegmentada(imagen_base_np, mascaras_memoria)

        # Miniaturas coloreadas
        miniaturas_coloreadas = [(m.miniatura, f"Máscara #{m.id}") for m in mascaras_memoria] #Guardamos las mascaras con su ID para el selector de mascaras       
        opciones_selector = [m.id for m in mascaras_memoria] #Guardamos el ID para los checkboxes

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

#Combina las máscaras segmentadas sobre la imagen original con sus colores asociados.
#Recibe:
# - imagen_np (np.ndarray): Imagen original en formato NumPy.
# - mascaras (List[MascaraSegmentada]): Lista de objetos con máscaras, color y miniatura.
#Devuelve:
# - Image.Image: Imagen con las máscaras superpuestas en sus colores.
def generar_imagen_con_mascaras_combinadas_mascarasegmentada(imagen_np: np.ndarray, mascaras: List[MascaraSegmentada]) -> Image.Image:
    try:
        overlay = imagen_np.copy()

        for m in mascaras:
            mask = m.binaria.astype(np.uint8)
            rgb_mask = np.zeros_like(imagen_np)
            for c in range(3):
                rgb_mask[:, :, c] = mask * m.color[c]
            overlay = cv2.addWeighted(overlay, 1.0, rgb_mask, 0.5, 0)

        return Image.fromarray(overlay)
    except Exception as e:
        print(f"Error generar mascaras combinadas mascarasegmentada: {e}")
        return Image.new("RGB", (512, 512), (255, 0, 0))

# Actualiza la imagen combinada al seleccionar máscaras desde el checkbox
def actualizar_mascaras_selector(indices: List[str]) -> Image.Image:
    if not indices or imagen_base_np is None:
        return Image.new("RGB", (512, 512), (0, 0, 0))

    seleccionados = list(map(int, indices))
    mascaras = [mascaras_memoria[i].binaria for i in seleccionados]
    colores = [mascaras_memoria[i].color for i in seleccionados]

    return generar_imagen_con_mascaras_combinadas(imagen_base_np, mascaras, colores)

# Esta función se llama cuando el usuario pulsa "Eliminar HUD"
# Toma las máscaras seleccionadas por el usuario y elimina esas regiones de la imagen
def lanzar_inpainting(metodo_inpainting: int, indices: list[str], filepath: str) -> Image.Image:    
    try:
        if filepath is None or not os.path.exists(filepath):
            return Image.new("RGB", (512, 512), (255, 0, 0))  # Imagen roja si algo va mal
        try:
            imagen = Image.open(filepath)   
        except Exception as e:
            return Image.new("RGB", (512, 512), (255, 0, 0))  # Imagen roja si algo va malImage.new("RGB", (512, 512), (255, 0, 0))  # Imagen roja si algo va mal

        # Validamos estado
        if not indices or not mascaras_memoria:
            return Image.new("RGB", (512, 512), (255, 0, 0))  # Imagen roja si algo va mal

        # Convertimos los índices a enteros
        seleccionados = list(map(int, indices))

        # Filtramos solo las máscaras elegidas
        mascaras_a_eliminar = [mascaras_memoria[i] for i in seleccionados]

        # Creamos el objeto de inpainting usando la factoría
        inpainter = InpaintingBase.crear(TipoInpainting(metodo_inpainting))

        # Cargamos el modelo si fuera necesario
        inpainter.cargar_modelo()

        # Ejecutamos la eliminación
        imagen_sin_hud = inpainter.eliminar_objetos(imagen, mascaras_a_eliminar)
        
    except Exception as e:
        print(f"❌ Error en inpainting: {e}")
        imagen_sin_hud = Image.new("RGB", (512, 512), (255, 0, 0))  # Imagen roja si algo va mal
        
    return imagen_sin_hud



# Esta función crea la interfaz gráfica con Gradio
def launch_interface():
    with gr.Blocks() as page:
        gr.Markdown(f"## {text_gradio_ui['titulo_app']}")

        with gr.Row(): 
                with gr.Column():
                # Cuadro para subida de imagen
                    with gr.Tab("Imágen original"):
                        image_input = gr.Image(type="filepath", label=f"{text_gradio_ui['subir_imagen']} {FORMATOS_TEXTO}")                         
                    with gr.Tab("Mascaras Combinadas"):
                        combined_mask_preview = gr.Image(label="🧵 Vista general de todas las segmentaciones", interactive=False)
                with gr.Column():
                    with gr.Tab("Mascaras"):
                        with gr.Column():
                            mascaras = gr.Gallery(label="🧩 Segmentos individuales", columns=4, rows=2, show_label=False)
                        with gr.Column():
                            segment_selector = gr.CheckboxGroup(label="🎯 Selecciona máscaras a mostrar", choices=[], interactive=True)                    
                    with gr.Tab("Imagen Final"):
                        image_output = gr.Image(label="✅ Imagen sin HUD", interactive=False)
                
                     
        
        with gr.Row(): 
            with gr.Column():
                # Cuadro con las propiedades de la imagen
                with gr.Tab(text_gradio_ui["propiedades_imagen"]):  
                    image_info = gr.Textbox(label="", lines=4, interactive=False) 
            with gr.Tab("Modelo de Segmentación"):
                with gr.Column():
                    opciones_sam = [("SAM B (rápido, poca precisión)", "vit_b"), ("SAM L (Equilibrado)", "vit_l"), ("SAM H (El más preciso)", "vit_h")]
                    sam_selector = gr.Radio(label="🧠 Modelo de SAM", choices=opciones_sam,  value="vit_b" ,interactive=True)
                with gr.Column():
                    opciones_segmentacion = [("Automático", 0), ("Punto", 1), ("Caja", 2), ("Pincel", 3)]
                    segmentation_selector = gr.Radio(label="🧠 Modelo de Segmentación", choices=opciones_segmentacion, value=0, interactive=True)
            with gr.Tab("Editor de imágen"):                
                with gr.Column():
                    opciones_inpainting = [("OpenCV", 0), ("LaMa", 1), ("Stable Diffusion", 2)]
                    inpaint_selector = gr.Radio(label="🧠 Modelo de Inpainting", choices=opciones_inpainting, value=0, interactive=True)                
        with gr.Row():
            segment_button = gr.Button("📐 Detectar Segmentos")
            apply_button = gr.Button("🧹 Eliminar HUD")  

        # Cada vez que se suba una imagen, se llamará a esta función para analizarla
        image_input.change(fn=get_image_properties, inputs=image_input, outputs=image_info)        
        # Al pulsar este boton, lanzamos el calculo de mascaras de segmentacion
        segment_button.click(fn=lanzar_segmentacion, inputs=[image_input, sam_selector, segmentation_selector], outputs=[mascaras, combined_mask_preview, segment_selector])
        # Cuando el usuario va marcando mascaras en el checkbox, se actualiza la imagen de mascaras combinadas
        segment_selector.change(fn=actualizar_mascaras_selector, inputs=segment_selector, outputs=combined_mask_preview)
        # Al pulsar el botón, se llama a la función de inpainting
        apply_button.click(fn=lanzar_inpainting, inputs=[inpaint_selector, segment_selector, image_input], outputs=image_output)
        # Registra los puntos que marca el usuario al hacer click en la imagen
        image_input.select(fn=registrar_puntos, inputs=None, outputs=None)


    #descomentar para produccion
    #page.launch()
    #debug
    page.launch(show_error=True, debug=True)
