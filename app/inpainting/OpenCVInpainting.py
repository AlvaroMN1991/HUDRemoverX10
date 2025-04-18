import cv2
import numpy as np
from app.inpainting.InpaintingBase import InpaintingBase
from app.utils.config import TipoInpainting
from app.sam.mascara_segmentada import MascaraSegmentada
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
    def eliminar_objetos(self, imagen: Image.Image, mascaras: List[MascaraSegmentada], prompt_sd: str="", negative_prompt_sd: str="") -> Image.Image:        
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
                
            # --- Mejora A: Recorte inteligente con margen ---
            coords = cv2.findNonZero(mascara_total)
            if coords is None:
                return imagen  # No hay nada que borrar

            x, y, w, h = cv2.boundingRect(coords)
            margen = 20  # píxeles de margen
            x1 = max(x - margen, 0)
            y1 = max(y - margen, 0)
            x2 = min(x + w + margen, anchura)
            y2 = min(y + h + margen, altura)

            # Recortamos la imagen y la máscara al área relevante
            recorte_imagen = imagen_np[y1:y2, x1:x2]
            recorte_mascara = mascara_total[y1:y2, x1:x2]

            # --- Mejora B: Suavizado de bordes ---
            recorte_mascara = cv2.GaussianBlur(recorte_mascara, (7, 7), sigmaX=2)

            # --- Mejora C: Escalado para interpolación ---
            escala = 2
            recorte_imagen_up = cv2.resize(recorte_imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
            recorte_mascara_up = cv2.resize(recorte_mascara, None, fx=escala, fy=escala, interpolation=cv2.INTER_NEAREST)

            # Inpainting en alta resolución
            inpainted_up = cv2.inpaint(recorte_imagen_up, recorte_mascara_up, 3, cv2.INPAINT_TELEA)

            # Redimensionamos a resolución original
            inpainted_down = cv2.resize(inpainted_up, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

            # Pegamos el parche de vuelta en la imagen original
            imagen_np[y1:y2, x1:x2] = inpainted_down

            # Convertimos la imagen resultante de NumPy a PIL para usarla en la interfaz
            return Image.fromarray(imagen_np)
        except Exception as e:
            print(f"Error en OpenCVInpainting: {e}")
            return Image.new("RGB", (512, 512), (255, 0, 0))  # Imagen roja si algo va mal

##################################################################################
#Version alternativa, funciona peor a pesar de tener mas ajustes
##################################################################################

    #Devuelve el algoritmo más adecuado según el tamaño de la zona a rellenar.
    def elegir_algoritmo_inpainting(self, mascara: np.ndarray) -> int:
        try:
            area = cv2.countNonZero(mascara)
            total_pixels = mascara.shape[0] * mascara.shape[1]

            if area / total_pixels > 0.1:
                return cv2.INPAINT_NS  # zona muy grande → mejor difuminar
            else:
                return cv2.INPAINT_TELEA  # zonas pequeñas → mejor preservar bordes
        except Exception as e:
            print(f"Error al elegir algoritmo de inpainting: {e}")
            return cv2.INPAINT_TELEA

    #Calcula el radio de inpainting de forma adaptativa en base al área de la máscara.
    def calcular_radio_adaptativo(self, mascara: np.ndarray, max_radio: int = 7, min_radio: int = 1) -> int:
        area = cv2.countNonZero(mascara)
        total = mascara.shape[0] * mascara.shape[1]
        proporción = area / total

        # Cuanto mayor la proporción → mayor el radio
        radio = int(min_radio + proporción * (max_radio - min_radio))
        return max(min_radio, min(radio, max_radio))  # lo limitamos

    #Elimina los objetos indicados por las máscaras usando el algoritmo de inpainting de OpenCV.
    #Recibe:
        # -imagen (np.ndarray): Imagen original en formato NumPy.
        # -mascaras (List[MascaraSegmentada]): Lista de objetos a eliminar.
    #Devuelve:
        # -Image.Image: Imagen con los objetos eliminados.
    def eliminar_objetos_pro(self, imagen: Image.Image, mascaras: List[MascaraSegmentada]) -> Image.Image:        
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
                
            # --- Mejora A: Recorte inteligente con margen ---
            coords = cv2.findNonZero(mascara_total)
            if coords is None:
                return imagen  # No hay nada que borrar

            # --- Recorte inteligente ---
            x, y, w, h = cv2.boundingRect(coords)
            margen = 20  # píxeles de margen
            x1 = max(x - margen, 0)
            y1 = max(y - margen, 0)
            x2 = min(x + w + margen, anchura)
            y2 = min(y + h + margen, altura)

            # Recortamos la imagen y la máscara al área relevante
            recorte_imagen = imagen_np[y1:y2, x1:x2]
            recorte_mascara = mascara_total[y1:y2, x1:x2]

             # --- Suavizado de bordes (A) ---
            kernel = np.ones((5, 5), np.uint8)
            mascara_suavizada = cv2.dilate(recorte_mascara, kernel, iterations=1)
            mascara_suavizada = cv2.GaussianBlur(mascara_suavizada, (5, 5), 0)
            _, mascara_binaria = cv2.threshold(mascara_suavizada, 10, 255, cv2.THRESH_BINARY)

            # --- División en zonas (C) ---
            zona_grande = cv2.erode(mascara_binaria, np.ones((7, 7), np.uint8), iterations=1)
            zona_detalle = cv2.subtract(mascara_binaria, zona_grande)

            # Inpainting paso 1          
            algoritmo_grande = self.elegir_algoritmo_inpainting(zona_grande)  # --- Selección de algoritmo (D) ---            
            radio_grande = self.calcular_radio_adaptativo(zona_grande) # --- Cálculo de radio dinámico (F) ---            
            paso1 = cv2.inpaint(recorte_imagen, zona_grande, inpaintRadius=radio_grande, flags=algoritmo_grande)

            # Inpainting paso 2
            radio_detalle = self.calcular_radio_adaptativo(zona_detalle)
            algoritmo_detalle = self.elegir_algoritmo_inpainting(zona_detalle)
            paso2 = cv2.inpaint(paso1, zona_detalle, inpaintRadius=radio_detalle, flags=algoritmo_detalle)

            # --- Blending final (E) ---
            feather = cv2.GaussianBlur(mascara_binaria.astype(np.float32), (21, 21), sigmaX=10) / 255.0
            feather = np.clip(feather[..., None], 0, 1)
            blended = (recorte_imagen * (1 - feather) + paso2 * feather).astype(np.uint8)

            # Sustituimos solo el área recortada
            imagen_np[y1:y2, x1:x2] = blended

            # Convertimos la imagen resultante de NumPy a PIL para usarla en la interfaz
            return Image.fromarray(imagen_np)
        except Exception as e:
            print(f"Error en OpenCVInpainting: {e}")
            return Image.new("RGB", (512, 512), (255, 0, 0))  # Imagen roja si algo va mal