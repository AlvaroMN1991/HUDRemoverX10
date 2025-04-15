from typing import List
import numpy as np
from PIL import Image
import cv2
from app.sam.mascara_segmentada import MascaraSegmentada

# Combina varias máscaras segmentadas en una imagen con colores superpuestos
def generar_imagen_con_mascaras_combinadas(imagen_np: np.ndarray, mascaras: List[MascaraSegmentada]) -> Image.Image:
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
        print(f"Error generando imagen con máscaras: {e}")
        return Image.new("RGB", (512, 512), (255, 0, 0))