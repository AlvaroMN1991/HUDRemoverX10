import cv2
import numpy as np
from app.inpainting.InpaintingBase import InpaintingBase
from app.utils.config import TipoInpainting
from app.sam.sam_loader import MascaraSegmentada
from typing import List
from PIL import Image

#Implementación del motor de inpainting usando OpenCV.
@InpaintingBase.register(TipoInpainting.OpenCV) #Esto es para que luego desde la interfaz, se pueda crear una clase de manera implicita
class OpenCVInpainting(InpaintingBase):
    
    
    tipo = TipoInpainting.OpenCV # Para la clase base

    def cargar_modelo(self):
        # OpenCV no necesita cargar modelo, así que lo dejamos vacío
        pass

    #Elimina los objetos indicados por las máscaras usando el algoritmo de inpainting de OpenCV.
    #Recibe:
        # -imagen (np.ndarray): Imagen original en formato NumPy.
        # -mascaras (List[MascaraSegmentada]): Lista de objetos a eliminar.
    #Devuelve:
        # -Image.Image: Imagen con los objetos eliminados.
    def eliminar_objetos(self, imagen: Image.Image, mascaras: List[MascaraSegmentada]) -> Image.Image:        
        try:
            # Convertimos la imagen PIL a un array NumPy
            imagen_np = np.array(imagen)  
            
            # Nos aseguramos de que la imagen esté en formato RGB uint8
            if imagen_np.ndim == 2:
                # Imagen en escala de grises → la convertimos a RGB
                imagen_np = np.stack([imagen_np] * 3, axis=-1)

            # Nos aseguramos de que la imagen está en el formato correcto (uint8, RGB)
            if imagen_np.dtype != np.uint8:
                try:
                    max_val = imagen_np.max()
                    if max_val <= 1.0:
                        imagen_np = (imagen_np * 255).astype(np.uint8)
                    else:
                        imagen_np = imagen_np.astype(np.uint8)
                except Exception as e:
                    raise ValueError(f"No se pudo normalizar imagen: {e}")
            
            # Creamos una máscara vacía del mismo tamaño que la imagen (1 canal)
            altura, anchura = imagen_np.shape[:2]
            mascara_total: np.ndarray = np.zeros((altura, anchura), dtype=np.uint8)

            # Combinamos todas las máscaras binarias en una sola máscara total (valor 255 donde hay que eliminar)
            for m in mascaras:
                # Nos aseguramos de que la máscara binaria sea del tipo correcto (uint8)
                binaria = m.binaria
                
                # Aseguramos tipo uint8 y rango correcto (0 o 255)
                if binaria.max() <= 1:
                    binaria = (binaria * 255).astype(np.uint8)
                else:
                    binaria = binaria.astype(np.uint8)

                # Redimensionamos si no coincide con la imagen
                if binaria.shape != mascara_total.shape:
                    binaria = cv2.resize(binaria, (anchura, altura), interpolation=cv2.INTER_NEAREST)

                # Hacemos que sea contigua en memoria para evitar errores en OpenCV
                binaria = np.ascontiguousarray(binaria)  # Evitamos errores de alineación de memoria

                # Combinamos esta máscara con la máscara total usando bitwise_or
                mascara_total = cv2.bitwise_or(mascara_total, binaria)        

            # Aplicamos el algoritmo de inpainting de OpenCV (telea es rápido y mantiene bordes)
            imagen_procesada = cv2.inpaint(imagen_np, mascara_total, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # Convertimos la imagen resultante de NumPy a PIL para usarla en la interfaz
            return Image.fromarray(imagen_procesada)
        except Exception as e:
            print(f"Error en OpenCVInpainting: {e}")
            return Image.new("RGB", (512, 512), (255, 0, 0))  # Imagen roja si algo va mal
