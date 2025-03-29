from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image
import numpy as np
import cv2
import random
import os
import urllib.request
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # type: ignore
from app.utils.tools import get_device_id
from app.utils.config import MODEL_URLS

# Clase para encapsular información de cada máscara
@dataclass
class MascaraSegmentada:
    binaria: np.ndarray                      # Máscara binaria (0 o 1)
    color: Tuple[int, int, int]              # Color RGB asignado
    miniatura: Image.Image                   # Imagen de vista previa coloreada
    sam_original: dict                       # Objeto original de SAM con metadatos

def descargar_modelo_si_no_existe(tipo_modelo: str, carpeta_modelos: str = "models") -> str:
    # Verifica y descarga el modelo si no existe
    os.makedirs(carpeta_modelos, exist_ok=True)
    nombre_fichero = os.path.basename(MODEL_URLS[tipo_modelo])
    ruta_local = os.path.join(carpeta_modelos, nombre_fichero)

    if not os.path.exists(ruta_local):
        print(f"📥 Descargando modelo {tipo_modelo}...")
        urllib.request.urlretrieve(MODEL_URLS[tipo_modelo], ruta_local)
        print("✅ Modelo descargado correctamente.")
    else:
        print(f"📁 Modelo {tipo_modelo} ya está disponible localmente.")

    return ruta_local


def cargar_sam_online(model_name: str):
    # Carga el modelo SAM desde el repositorio oficial
    try:
        checkpoint_path = descargar_modelo_si_no_existe(model_name)
        modelo = sam_model_registry[model_name](checkpoint=checkpoint_path)
        device = get_device_id()
        modelo.to(device)
        modelo.eval()
        return modelo
    except Exception as e:
        print(f"❌ Error al cargar modelo SAM: {e}")
        raise

# Función para calcular la distancia desde el origen (0,0) para ordenar las máscaras
# Utiliza la esquina superior izquierda del bounding box
def distancia_desde_origen(mask: dict) -> float:
    try:
        x_min, y_min, _, _ = mask["bbox"]
        return np.sqrt(x_min**2 + y_min**2)
    except Exception as e:
        print(f"Error calculando distancia desde origen: {e}")
        return float("inf")

# Usa SAM para segmentar la imagen automáticamente y devolver las máscaras
# Devuelve:
# - Lista de objetos MascaraSegmentada con toda la info útil para postprocesado e interfaz
# - Imagen combinada con todas las máscaras coloreadas
def segmentar_automaticamente(imagen_pil: Image.Image, modelo_sam) -> tuple[List[MascaraSegmentada], Image.Image]:
    try:
        # Convertimos la imagen a NumPy RGB
        imagen_np = np.array(imagen_pil.convert("RGB"))

        # Inicializamos el generador automático de máscaras
        generator = SamAutomaticMaskGenerator(modelo_sam)

        # Generamos las máscaras
        masks = generator.generate(imagen_np)

        # Si no se detectaron máscaras, devolvemos vacío
        if not masks:
            return [], imagen_pil

        # Ordenamos las máscaras desde la esquina superior izquierda
        masks = sorted(masks, key=distancia_desde_origen)

        # Inicializamos lista de objetos de máscara y la imagen overlay
        objetos_mascaras: List[MascaraSegmentada] = []
        overlay = imagen_np.copy()

        # Recorremos cada máscara generada
        for mask in masks:
            try:
                seg = mask["segmentation"].astype(np.uint8)  # binaria 0/1

                # Generamos un color aleatorio y lo convertimos a tupla (R, G, B)
                color: tuple[int, int, int] = tuple(np.random.randint(0, 255, size=3, dtype=int))

                # Creamos la máscara RGB del mismo tamaño
                rgb_mask = np.zeros_like(imagen_np)
                for c in range(3):
                    rgb_mask[:, :, c] = seg * color[c]

                # Generamos la miniatura con transparencia
                blended = cv2.addWeighted(imagen_np, 0.7, rgb_mask, 0.5, 0)
                miniatura_pil = Image.fromarray(blended)

                # Añadimos al overlay combinado
                overlay = cv2.addWeighted(overlay, 1.0, rgb_mask, 0.5, 0)

                # Guardamos el objeto completo con máscara, color, miniatura y metadatos
                objetos_mascaras.append(MascaraSegmentada(binaria=seg, color=color, miniatura=miniatura_pil,sam_original=mask))
            
            except Exception as e:
                print(f"Error procesando una máscara: {e}")
                continue

        # Convertimos la imagen combinada a PIL
        imagen_combinada = Image.fromarray(overlay)

        return objetos_mascaras, imagen_combinada

    except Exception as e:
        print(f"Error general en segmentar_automaticamente: {e}")
        return [], imagen_pil
