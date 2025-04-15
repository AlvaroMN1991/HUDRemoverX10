from PIL import Image
import numpy as np
import gradio as gr
from app.utils.config import TipoSegmentacion
from app.controllers.utils_mascaras import generar_imagen_con_mascaras_combinadas
from app.globals.estado_global import EstadoGlobal 

#Este método muestra la imagen en la interfaz de usuario
def cargar_y_mostrar_imagen(file, modo_actual: int, estado: EstadoGlobal):
    try:
        if file is None or not file.name:
            return Image.new("RGB", (512, 512), (64, 64, 64))
        imagen = Image.open(file.name).convert("RGB")
        estado.imagen_base_np = np.array(imagen)
        estado.imagen_pil = imagen
        return gr.update(value=imagen, visible=True, interactive=(modo_actual == TipoSegmentacion.Punto.value))
    except Exception as e:
        print(f"❌ Error cargar imagen: {e}")
        return Image.new("RGB", (512, 512), (64, 64, 64))

#Actualiza las imagenes añadiendo las mascaras.
def actualizar_imagen_mascaras(indices: list[str], estado: EstadoGlobal) -> Image.Image:
    try:
        if not indices or estado.imagen_base_np is None:
            return Image.new("RGB", (512, 512), (0, 0, 0))

        seleccionados = list(map(int, indices))
        mascaras = [estado.mascaras_memoria[i] for i in seleccionados]

        return generar_imagen_con_mascaras_combinadas(estado.imagen_base_np, mascaras)
    except Exception as e:
        print(f"❌ Error actualizar mascaras: {e}")
        return Image.new("RGB", (512, 512), (255, 0, 0))
