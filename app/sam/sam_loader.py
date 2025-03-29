from transformers import SamModel, SamProcessor #type:ignore
from app.utils.tools import get_device_id
from PIL import Image
from app.utils.config import MODEL_URLS
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # type: ignore
import numpy as np
from matplotlib import cm
import os
import urllib.request
import random
import cv2
import numpy as np


#Carga el modelo indicado por el usuario en la interfaz
def cargar_sam_online(model_name: str):
    checkpoint_path = descargar_modelo_si_no_existe(model_name)
    modelo = sam_model_registry[model_name](checkpoint=checkpoint_path)
    device = get_device_id()  # Esto debería devolver "cuda" o "cpu"
    modelo.to(device)
    modelo.eval()
    return modelo

def aplicar_colormap(mask: np.ndarray) -> Image.Image:
    color_array = cm.viridis(mask / 255.0)[:, :, :3]  #type:ignore # Normaliza y aplica colormap
    color_array = (color_array * 255).astype(np.uint8)
    return Image.fromarray(color_array)

#Usa SAM para segmentar la imagen automáticamente y devuelve las máscaras
def segmentar_automaticamente(imagen_pil: Image.Image, modelo_sam) -> tuple[list[Image.Image], Image.Image]:
    imagen_np = np.array(imagen_pil.convert("RGB"))
    generator = SamAutomaticMaskGenerator(modelo_sam)
    masks = generator.generate(imagen_np)

    if not masks:
        return [], imagen_pil

    # Creamos la imagen combinada con todas las máscaras coloreadas
    overlay = imagen_np.copy()

    for mask in masks:
        color = [random.randint(0, 255) for _ in range(3)]
        mask_array = mask["segmentation"].astype(np.uint8) * 255

        # Crear máscara 3 canales
        mask_3c = np.stack([mask_array]*3, axis=-1)

        # Colorear solo donde la máscara es 1
        colored_mask = np.zeros_like(overlay)
        for i in range(3):
            colored_mask[..., i] = color[i]
        masked = cv2.bitwise_and(colored_mask, mask_3c)

        # Combinamos con la imagen original
        overlay = cv2.addWeighted(overlay, 1.0, masked, 0.5, 0)

    # Convertimos a PIL
    imagen_combinada = Image.fromarray(overlay)

    # Lista de imágenes individuales (para mantener compatibilidad con la interfaz)
    imagenes_mascaras: list[Image.Image] = []
    for mask in masks:
        binaria = (mask["segmentation"].astype(np.uint8)) * 255
        imagenes_mascaras.append(Image.fromarray(binaria))

    return imagenes_mascaras, imagen_combinada

def descargar_modelo_si_no_existe(tipo_modelo: str, carpeta_modelos: str = "models") -> str:
    os.makedirs(carpeta_modelos, exist_ok=True)
    nombre_fichero = os.path.basename(MODEL_URLS[tipo_modelo])
    ruta_local = os.path.join(carpeta_modelos, nombre_fichero)
    print(ruta_local)
    print(nombre_fichero)
    
    if not os.path.exists(ruta_local):
        print(f"📥 Descargando modelo {tipo_modelo}...")
        urllib.request.urlretrieve(MODEL_URLS[tipo_modelo], ruta_local)
        print("✅ Modelo descargado correctamente.")
    else:
        print(f"📁 Modelo {tipo_modelo} ya está disponible localmente.")

    return ruta_local