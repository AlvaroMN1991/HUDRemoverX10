from typing import List, Tuple, Any
from PIL import Image
import numpy as np
import cv2
import gradio as gr
from app.globals.estado_global import EstadoGlobal

# Gestiona los clics del usuario sobre la imagen
def registrar_punto(evt: gr.SelectData, metodo_segmentacion: int, estado: EstadoGlobal) -> Tuple[List[List[int]], Any, Image.Image]:
    try:
        #if evt is None or getattr(evt, "value", None) is None or estado.imagen_base_np is None:
         #   return [], gr.update(choices=[], value=[]), Image.new("RGB", (512, 512), (64, 64, 64))

        #if metodo_segmentacion != 1:
        #    return [], gr.update(choices=[], value=[]), Image.fromarray(estado.imagen_base_np.copy())

        punto = (evt.value[0], evt.value[1])
        print(f"Coordenadas recibidas: {punto}")
        estado.puntos_usuario.append(punto)
        print(f"Puntos acumulados: {estado.puntos_usuario}")

        # Dibujar puntos
        imagen = estado.imagen_base_np.copy()
        print(f"Shape de imagen: {imagen.shape}")
        for x, y in estado.puntos_usuario:
            print(f"Dibujando círculo en: ({x}, {y})")
            cv2.circle(imagen, (x, y), 10, (0, 255, 0), -1)

        datos = [list(p) for p in estado.puntos_usuario]
        indices = list(map(str, range(len(estado.puntos_usuario))))
        return datos, gr.update(choices=indices, value=[]), Image.fromarray(imagen)

    except Exception as e:
        print(f"❌ Error registrar_punto: {e}")
        print(f"Debug evt: {vars(evt) if evt is not None else 'evt is None'}")
        return [], gr.update(choices=[], value=[]), Image.new("RGB", (512, 512), (255, 0, 0))
        
# Elimina puntos seleccionados por el usuario
def eliminar_puntos(indices_seleccionados: List[str], estado: EstadoGlobal) -> Tuple[List[List[int]], Any, Image.Image]:
    try:
        for idx in sorted(map(int, indices_seleccionados), reverse=True):
            if 0 <= idx < len(estado.puntos_usuario):
                estado.puntos_usuario.pop(idx)

        imagen = estado.imagen_base_np.copy() if estado.imagen_base_np is not None else np.zeros((512,512,3), dtype=np.uint8)
        for x, y in estado.puntos_usuario:
            cv2.circle(imagen, (x, y), 30, (255, 51, 51), -1)  # radio 30, color rojo claro

        datos = [list(p) for p in estado.puntos_usuario]
        opciones = list(map(str, range(len(estado.puntos_usuario))))
        return datos, gr.update(choices=opciones, value=[]), Image.fromarray(imagen)

    except Exception as e:
        print(f"❌ Error eliminar_puntos: {e}")
        return [], gr.update(choices=[], value=[]), Image.new("RGB", (512, 512), (255, 0, 0))
