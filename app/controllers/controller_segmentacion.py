from typing import Tuple, List, Any
from PIL import Image
import numpy as np
import gradio as gr
from app.globals.estado_global import EstadoGlobal
from app.sam.SegmentadorBase import SegmentadorBase, Segmentacion
from app.utils.config import TipoSegmentacion
from app.sam.sam_loader import cargar_sam_online, segmentar_automaticamente
from app.controllers.utils_mascaras import generar_imagen_con_mascaras_combinadas

# Ejecuta la segmentación en función del tipo elegido por el usuario
def ejecutar_segmentacion(imagen: Image.Image, modelo_clave: str, tipo_segmentacion: int, estado: EstadoGlobal) -> Tuple[List[Tuple[Image.Image, str]], Image.Image, Any]:
    try:
        estado.imagen_pil = imagen
        estado.imagen_base_np = np.array(imagen.convert("RGB"))

        modelo = cargar_sam_online(modelo_clave, modo=tipo_segmentacion)

        if tipo_segmentacion == TipoSegmentacion.Automatico.value:
            estado.mascaras_memoria, imagen_combinada = segmentar_automaticamente(imagen, modelo)
        else:
            anotacion = Segmentacion(tipo=TipoSegmentacion(tipo_segmentacion), puntos=estado.puntos_usuario, etiquetas=[1]*len(estado.puntos_usuario))
            segmentador = SegmentadorBase.crear(TipoSegmentacion(tipo_segmentacion), modelo)
            estado.mascaras_memoria = segmentador.segmentar(estado.imagen_base_np, anotacion)
            imagen_combinada = generar_imagen_con_mascaras_combinadas(estado.imagen_base_np, estado.mascaras_memoria)

        miniaturas = [(m.miniatura, f"Máscara #{m.id}") for m in estado.mascaras_memoria]
        opciones = [m.id for m in estado.mascaras_memoria]

        return miniaturas, imagen_combinada, gr.update(choices=opciones, value=[])

    except Exception as e:
        print(f"❌ Error en ejecutar_segmentacion: {e}")
        return [], Image.new("RGB", (512, 512), (255, 0, 0)), gr.update(choices=[], value=[])