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

# Ordenamos las máscaras según la distancia desde (0,0) a la esquina superior izquierda de su bbox
# Calculamos la distancia euclidiana ??
def distancia_desde_origen(mask):
    x_min, y_min, _, _ = mask["bbox"]
    return np.sqrt(x_min**2 + y_min**2)

#Usa SAM para segmentar la imagen automáticamente y devuelve las máscaras
def segmentar_automaticamente(imagen_pil: Image.Image, modelo_sam) -> tuple[list[Image.Image], Image.Image]:
    imagen_np = np.array(imagen_pil.convert("RGB"))
    generator = SamAutomaticMaskGenerator(modelo_sam)
    masks = generator.generate(imagen_np)

    if not masks:
        return [], imagen_pil    

    #Ordenamos las mascaras para que visualmente esten colocadas partiendo desde la esquina superior izquierda
    masks = sorted(masks, key=distancia_desde_origen)

    # Copia para la imagen combinada con todas las máscaras
    overlay = imagen_np.copy()
    imagenes_mascaras: list[Image.Image] = []

    for mask in masks:
        # Generamos un color aleatorio por máscara
        color = [random.randint(0, 255) for _ in range(3)]

        # Convertimos la máscara binaria (bool) a uint8 (0 o 255)
        mask_array = mask["segmentation"].astype(np.uint8) * 255

        # Creamos una versión 3 canales de la máscara para aplicar color
        mask_3c = np.stack([mask_array] * 3, axis=-1)

        # Creamos imagen coloreada
        colored_mask = np.zeros_like(imagen_np)
        for i in range(3):
            colored_mask[..., i] = color[i]

        # Aplicamos la máscara al color
        masked = cv2.bitwise_and(colored_mask, mask_3c)

        # Combinamos con la imagen base con transparencia
        blended = cv2.addWeighted(imagen_np, 0.7, masked, 0.3, 0)

        # Añadimos a la lista de miniaturas
        imagenes_mascaras.append(Image.fromarray(blended))

        # También la sumamos al overlay combinado
        overlay = cv2.addWeighted(overlay, 1.0, masked, 0.5, 0)

    # Convertimos overlay combinado a PIL
    imagen_combinada = Image.fromarray(overlay)
    

    return imagenes_mascaras, imagen_combinada



#Devuelve la lista de imágenes de máscaras coloreadas + la imagen combinada.
def generar_mascaras_coloreadas(imagen_np: np.ndarray, masks: list[dict]) -> tuple[list[Image.Image], Image.Image]:

    overlay_images = []
    combined_overlay = imagen_np.copy()

    for mask in masks:
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        seg = mask["segmentation"].astype(np.uint8)

        mask_rgb = np.zeros_like(imagen_np)
        for c in range(3):
            mask_rgb[:, :, c] = seg * color[c]

        blended = cv2.addWeighted(imagen_np, 0.7, mask_rgb, 0.5, 0)
        overlay_images.append(Image.fromarray(blended))

        combined_overlay = cv2.add(combined_overlay, mask_rgb)

    imagen_combined = Image.fromarray(combined_overlay)
    overlay_images.insert(0, imagen_combined)

    return overlay_images, imagen_combined
