from app.sam.SegmentadorBase import SegmentadorBase, Segmentacion
from app.utils.config import TipoSegmentacion
from app.sam.mascara_segmentada import MascaraSegmentada
from PIL import Image
from typing import List, Optional, Tuple
from segment_anything import SamPredictor #type: ignore
import numpy as np
import cv2

@SegmentadorBase.register(TipoSegmentacion.Caja) #Esto es para que luego desde la interfaz, se pueda crear una clase de manera implicita
class SegmentadorPorCaja(SegmentadorBase):

    tipo = TipoSegmentacion.Caja # Para la clase base

    def segmentar(self, imagen: np.ndarray, segmentacion: Segmentacion) -> List[MascaraSegmentada]:
        try:
            if segmentacion.tipo != TipoSegmentacion.Caja:
                raise ValueError("Anotación incorrecta para segmentación por caja.")

            if len(segmentacion.puntos) != 2:
                raise ValueError("La segmentación por caja requiere exactamente 2 puntos")

            # Convertir imagen a formato correcto si no es np.ndarray RGB
            if isinstance(imagen, Image.Image):
                imagen = np.array(imagen.convert("RGB"))
            elif isinstance(imagen, np.ndarray):
                if imagen.ndim == 2:  # escala de grises
                    imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
                elif imagen.shape[2] == 4:  # RGBA
                    imagen = imagen[:, :, :3]
                imagen = imagen.astype(np.uint8)

            # Extraemos la caja correctamente ordenada
            (x0, y0), (x1, y1) = segmentacion.puntos
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])
            input_box = np.array([[x_min, y_min, x_max, y_max]])

            # Configuramos el predictor
            predictor = SamPredictor(self.predictor)
            predictor.is_image_set = False
            predictor.set_image(imagen)

            # Generamos las máscaras desde la caja
            masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)

            if not masks.any():
                print("⚠️ No se generaron máscaras desde la caja")
                return []

            # Inicializamos lista de objetos de máscara y la imagen overlay
            mascaras_segmentadas: List[MascaraSegmentada] = []
            overlay = imagen.copy()

            # Procesamos cada máscara
            for idx, mask in enumerate(masks):
                try:
                    seg = mask.astype(np.uint8)

                    color: tuple[int, int, int] = tuple(np.random.randint(0, 255, size=3, dtype=int))

                    rgb_mask = np.zeros_like(imagen)
                    for c in range(3):
                        rgb_mask[:, :, c] = seg * color[c]

                    blended = cv2.addWeighted(imagen, 0.7, rgb_mask, 0.5, 0)
                    miniatura_pil = Image.fromarray(blended)

                    overlay = cv2.addWeighted(overlay, 1.0, rgb_mask, 0.5, 0)

                    mascaras_segmentadas.append(MascaraSegmentada(
                        binaria=seg,
                        color=color,
                        miniatura=miniatura_pil,
                        sam_original=mask,
                        id=idx
                    ))
                except Exception as e:
                    print(f"Error procesando una máscara de caja: {e}")
                    continue

            return mascaras_segmentadas

        except Exception as e:
            print(f"❌ Error al segmentar Caja: {e}")
            return []


    #Dibuja una caja roja entre dos puntos sobre una imagen NumPy.
    #Devuelve una imagen PIL con la caja dibujada.
    def dibujar(self, imagen_np: Optional[np.ndarray], puntos: List[Tuple[int, int]]) -> Image.Image:
        if imagen_np is None or len(puntos) != 2:
            return Image.fromarray(imagen_np) if imagen_np is not None else Image.new("RGB", (512, 512), (0, 0, 0))

        try:
            imagen = imagen_np.copy()
            (x1, y1), (x2, y2) = puntos
            esquina1 = (min(x1, x2), min(y1, y2))
            esquina2 = (max(x1, x2), max(y1, y2))

            cv2.rectangle(imagen, esquina1, esquina2, color=(0, 0, 255), thickness=2)

            return Image.fromarray(imagen)
        except Exception as e:
            print(f"❌ Error al dibujar caja: {e}")
            return Image.new("RGB", (512, 512), (255, 0, 0))