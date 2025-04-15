from app.sam.SegmentadorBase import SegmentadorBase, Segmentacion
from app.utils.config import TipoSegmentacion
from app.sam.mascara_segmentada import MascaraSegmentada
from PIL import Image
from typing import List
from segment_anything import SamPredictor #type: ignore
import numpy as np
import cv2

@SegmentadorBase.register(TipoSegmentacion.Punto) #Esto es para que luego desde la interfaz, se pueda crear una clase de manera implicita
class SegmentadorPorPunto(SegmentadorBase):

    tipo = TipoSegmentacion.Punto # Para la clase base

    def segmentar(self, imagen: np.ndarray, segmentacion: Segmentacion) -> List[MascaraSegmentada]:
        try:
            predictor = None

            if segmentacion.tipo != TipoSegmentacion.Punto:
                raise ValueError("Anotación incorrecta para segmentación por punto.")

            if isinstance(imagen, Image.Image):
                imagen = np.array(imagen.convert("RGB"))
            elif isinstance(imagen, np.ndarray):
                if imagen.ndim == 2:  # escala de grises
                    imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
                elif imagen.shape[2] == 4:  # RGBA
                    imagen = imagen[:, :, :3]
                imagen = imagen.astype(np.uint8)

            puntos_np = np.array(segmentacion.puntos)
            etiquetas_np = np.array(segmentacion.etiquetas)

            # Inicializamos el generador automático de máscaras
            predictor = SamPredictor(self.predictor)
            print(f"📷 Imagen shape: {imagen.shape}, dtype: {imagen.dtype}")
            predictor.is_image_set = False
            predictor.set_image(imagen)

            # Validación defensiva
            if not segmentacion.puntos:
                raise ValueError("No se han proporcionado puntos para segmentación por puntos")

            puntos_np = np.array(segmentacion.puntos, dtype=np.float32)
            print("📍 Puntos:", puntos_np)
            
            # 💡 Asegura que puntos_np.shape == (N, 2) y etiquetas_np.shape == (N,)
            if puntos_np.ndim != 2 or puntos_np.shape[1] != 2:
                raise ValueError(f"Formato incorrecto de puntos: {puntos_np.shape}")
           
            masks, scores, _ = predictor.predict(point_coords=puntos_np, point_labels=etiquetas_np, multimask_output=True)

            # 🔥 Filtro por score (0.7 por defecto)
            mascaras_filtradas = [m for m, s in zip(masks, scores) if s >= 0.8]

            # Asegurar al menos una máscara
            if not mascaras_filtradas:
                mascaras_filtradas = [masks[np.argmax(scores)]]

            # Inicializamos lista de objetos de máscara y la imagen overlay
            mascaras_segmentadas: List[MascaraSegmentada] = []
            overlay = imagen.copy()

            # Recorremos cada máscara generada
            for idx, mask in enumerate(masks):
                try:
                    seg = mask.astype(np.uint8)

                    # Generamos un color aleatorio y lo convertimos a tupla (R, G, B)
                    color: tuple[int, int, int] = tuple(np.random.randint(0, 255, size=3, dtype=int))

                    # Creamos la máscara RGB del mismo tamaño
                    rgb_mask = np.zeros_like(imagen)
                    for c in range(3):
                        rgb_mask[:, :, c] = seg * color[c]

                    # Generamos la miniatura con transparencia
                    blended = cv2.addWeighted(imagen, 0.7, rgb_mask, 0.5, 0)
                    miniatura_pil = Image.fromarray(blended)

                    # Añadimos al overlay combinado
                    overlay = cv2.addWeighted(overlay, 1.0, rgb_mask, 0.5, 0)

                    # Guardamos el objeto completo con máscara, color, miniatura y metadatos
                    mascaras_segmentadas.append(MascaraSegmentada(binaria=seg, color=color, miniatura=miniatura_pil,sam_original=mask, id=idx))
                
                except Exception as e:
                    print(f"Error procesando una máscara: {e}")
                    continue

            return mascaras_segmentadas
        except Exception as e:
            print(f"❌ Error al segmentar Punto: {e}")
            return []
