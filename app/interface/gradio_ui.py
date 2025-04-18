import gradio as gr
from app.lang.es import text_gradio_ui
from app.utils.config import FORMATOS_TEXTO, TipoSegmentacion
from app.utils.tools import obtener_propiedades_imagen

# Controladores
from app.controllers.controller_puntos import registrar_punto, eliminar_puntos
from app.controllers.controller_segmentacion import ejecutar_segmentacion
from app.controllers.controller_inpainting import ejecutar_inpainting
from app.controllers.controller_image import cargar_y_mostrar_imagen, actualizar_imagen_mascaras

def safe_callback(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"Error en callback: {e}")
            # Opcional: podrías retornar gr.update o un mensaje de error para la UI
    return wrapper


class GradioInterface:
    def __init__(self, estado):
        self.estado = estado
        
    #Construccion de la interfaz de usuario
    def build(self):
        with gr.Blocks() as page:
            gr.Markdown(f"## {text_gradio_ui['titulo_app']}")

            with gr.Row():
                with gr.Column():
                    with gr.Tab("Imágen original"):
                        selector_archivo = gr.File(label=f"{text_gradio_ui['subir_imagen']} {FORMATOS_TEXTO}", file_types=[".png", ".jpg", ".jpeg"])
                        image_input = gr.Image(type="pil", label=f"{text_gradio_ui['preview_imagen']}", visible=False, interactive=False)
                    with gr.Tab("Mascaras Combinadas"):
                        combined_mask_preview = gr.Image(label="🧵 Vista general de todas las segmentaciones", interactive=False)
                    #with gr.Tab("Puntos seleccionados"):
                    #    tabla_puntos = gr.Dataframe(headers=["X", "Y"], interactive=True, visible=True, label="📍 Puntos marcados")
                    #    selector_filas = gr.CheckboxGroup(label="Puntos generados", choices=[], interactive=True)
                    #    boton_eliminar_puntos = gr.Button("❌ Eliminar seleccionados")

                with gr.Column():
                    with gr.Tab("Mascaras"):
                        mascaras = gr.Gallery(label="🧩 Segmentos individuales", columns=4, rows=2, show_label=False)
                        segment_selector = gr.CheckboxGroup(label="🎯 Selecciona máscaras a mostrar", choices=[], interactive=True)
                    with gr.Tab("Imagen Final"):
                        image_output = gr.Image(label="✅ Imagen sin HUD", interactive=False)

            with gr.Row():
                with gr.Column():
                    with gr.Tab(text_gradio_ui["propiedades_imagen"]):
                        image_info = gr.Textbox(label="", lines=4, interactive=False)
                with gr.Tab("Modelo de Segmentación"):
                    opciones_sam = [("SAM B (rápido, poca precisión)", "vit_b"), ("SAM L (Equilibrado)", "vit_l"), ("SAM H (El más preciso)", "vit_h")]                    
                    sam_selector = gr.Radio(label="🧠 Modelo de SAM", choices=opciones_sam, value="vit_b", interactive=True)
                    opciones_inpainting = [("OpenCV", 0), ("LaMa", 1), ("Stable Diffusion", 2)]
                    inpaint_selector = gr.Radio(label="🧠 Modelo de Inpainting", choices=opciones_inpainting, value=0, interactive=True)
                    prompt_sd = gr.Textbox(label="Prompt para Stable Diffusion", placeholder="Introduce tu prompt aquí...", interactive=False, visible=False)  
                    negative_prompt_sd = gr.Textbox(label="Negative Prompt para Stable Diffusion", placeholder="Introduce tu prompt aquí...", interactive=False, visible=False)              
                #with gr.Tab("Editor de imágen"):
                    opciones_segmentacion = [("Automático", 0), ("Punto", 1), ("Caja", 2), ("Pincel", 3)]
                    segmentation_selector = gr.Radio(label="🧠 Modelo de Segmentación", choices=opciones_segmentacion, value=0, interactive=True, visible=False)
            with gr.Row():
                segment_button = gr.Button("📐 Detectar Segmentos")
                apply_button = gr.Button("🧹 Eliminar HUD")


            #segmentation_selector.change(fn=self._actualizar_interaccion, inputs=segmentation_selector, outputs=image_input)
            selector_archivo.change(fn=self.on_cargar_y_mostrar_imagen, inputs=[selector_archivo, segmentation_selector], outputs=image_input)
            image_input.change(fn=obtener_propiedades_imagen, inputs=[image_input, selector_archivo], outputs=image_info)
            segment_button.click(fn=self.on_ejecutar_segmentacion, inputs=[image_input, sam_selector, segmentation_selector], outputs=[mascaras, combined_mask_preview, segment_selector])
            segment_selector.change(fn=self.on_actualizar_imagen_mascaras, inputs=segment_selector, outputs=combined_mask_preview)
            apply_button.click(fn=self.on_ejecutar_inpainting, inputs=[inpaint_selector, segment_selector, prompt_sd, negative_prompt_sd], outputs=image_output)
            #image_input.select(fn=self.on_registrar_punto, inputs=[image_input, segmentation_selector], outputs=[tabla_puntos, selector_filas, image_input])
            #boton_eliminar_puntos.click(fn=self.on_eliminar_puntos, inputs=[selector_filas], outputs=[tabla_puntos, selector_filas, image_input])
            inpaint_selector.change(fn=self.actualizar_prompt_sd, inputs=inpaint_selector, outputs=[prompt_sd, negative_prompt_sd])
        return page
        
    #Funciones de eventos
    @safe_callback
    def _actualizar_interaccion(self, tipo):
        return gr.update(interactive=(tipo == TipoSegmentacion.Punto.value))

    @safe_callback
    def on_cargar_y_mostrar_imagen(self, file, modo_actual):
        return cargar_y_mostrar_imagen(file, modo_actual, self.estado)

    @safe_callback
    def on_actualizar_imagen_mascaras(self, indices):
        return actualizar_imagen_mascaras(indices, self.estado)

    @safe_callback
    def on_ejecutar_segmentacion(self, img, model, tipo):    
        return ejecutar_segmentacion(img, model, tipo, self.estado)
    
    @safe_callback
    def on_ejecutar_inpainting(self, metodo, indices, prompt_sd, negative_prompt_sd):
        return ejecutar_inpainting(metodo, indices, prompt_sd, negative_prompt_sd, self.estado)

    @safe_callback
    def on_registrar_punto(self, evt, tipo):
        return registrar_punto(evt, tipo, self.estado)

    @safe_callback
    def on_eliminar_puntos(self, sel):
        return eliminar_puntos(sel, self.estado)

    @safe_callback
    def actualizar_prompt_sd(self,modelo):
        # El valor 2 corresponde a "Stable Diffusion"
        if modelo==2:
            return gr.update(interactive=True, visible=True), gr.update(interactive=True, visible=True)
        else:
            return gr.update(interactive=False, visible=False), gr.update(interactive=False, visible=False)
        
