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
from app.utils.config import MODEL_URLS, TipoSegmentacion

# Clase para encapsular información de cada máscara
@dataclass
class MascaraSegmentada:
    binaria: np.ndarray                      # Máscara binaria (0 o 1)
    color: Tuple[int, int, int]              # Color RGB asignado
    miniatura: Image.Image                   # Imagen de vista previa coloreada
    sam_original: dict                       # Objeto original de SAM con metadatos
    id:int = -1                              # ID de la máscara (opcional, para referencia)

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


def cargar_sam_online(model_name: str, modo:int):
    # Carga el modelo SAM desde el repositorio oficial
    try:
        checkpoint_path = descargar_modelo_si_no_existe(model_name)
        modelo = sam_model_registry[model_name](checkpoint=checkpoint_path)
        device = get_device_id()
        modelo.to(device)
        modelo.eval()
        if TipoSegmentacion(modo) == TipoSegmentacion.Automatico:
            return SamAutomaticMaskGenerator(modelo)
        else:
            return modelo  # devuelves el modelo crudo, para usar con SamPredictor
      
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


#Elimina máscaras cuya área sea menor que el umbral dado. De momento no usamos
#Recibe:
# - mascaras (List[np.ndarray]): Lista de máscaras binarias (0-255).
# - umbral_pixeles (int): Área mínima permitida para conservar la máscara.
#Decuelve:
# - List[np.ndarray]: Lista filtrada de máscaras.
def filtrar_mascaras_pequenas(mascaras: List[np.ndarray], umbral_pixeles: int = 500) -> List[np.ndarray]:
    try:
        mascaras_filtradas = []
        for m in mascaras:
            mask_array = m["segmentation"]
            area = mask_array.sum() if mask_array.max() <= 1 else cv2.countNonZero(mask_array.astype(np.uint8))
            if area >= umbral_pixeles:
                mascaras_filtradas.append(m)
        
        return mascaras_filtradas      
    except Exception as e:
        print(f"❌ Error al filtrar mascaras en SAM: {e}")
        return mascaras


# Usa SAM para segmentar la imagen automáticamente y devolver las máscaras
# Devuelve:
# - Lista de objetos MascaraSegmentada con toda la info útil para postprocesado e interfaz
# - Imagen combinada con todas las máscaras coloreadas
def segmentar_automaticamente(imagen_pil: Image.Image, modelo_sam) -> tuple[List[MascaraSegmentada], Image.Image]:
    try:
        # Convertimos la imagen a NumPy RGB
        imagen_np = np.array(imagen_pil.convert("RGB"))

        # Inicializamos el generador automático de máscaras
        #generator = SamAutomaticMaskGenerator(modelo_sam)
        generator = modelo_sam
        
        # Generamos las máscaras
        masks = generator.generate(imagen_np)

        # Si no se detectaron máscaras, devolvemos vacío
        if not masks:
            return [], imagen_pil
    
        mascaras_filtradas = []
        for m in masks:
            if m["predicted_iou"] >= 0.9:
                mascaras_filtradas.append(m) 

        if not mascaras_filtradas:
            print("⚠️ SAM no encontró máscaras con score suficiente. Usando todas.")
            mascaras_filtradas = masks
        else:
            masks = mascaras_filtradas

        # Ordenamos las máscaras desde la esquina superior izquierda
        masks = sorted(masks, key=distancia_desde_origen)

        #masks = filtrar_mascaras_pequenas(masks, umbral_pixeles=600)

        # Inicializamos lista de objetos de máscara y la imagen overlay
        objetos_mascaras: List[MascaraSegmentada] = []
        overlay = imagen_np.copy()

        # Recorremos cada máscara generada
        for idx, mask in enumerate(masks):
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
                objetos_mascaras.append(MascaraSegmentada(binaria=seg, color=color, miniatura=miniatura_pil,sam_original=mask, id=idx))
            
            except Exception as e:
                print(f"Error procesando una máscara: {e}")
                continue

        # Convertimos la imagen combinada a PIL
        imagen_combinada = Image.fromarray(overlay)

        return objetos_mascaras, imagen_combinada

    except Exception as e:
        print(f"Error general en segmentar_automaticamente: {e}")
        return [], imagen_pil


#Mejora el contraste y nitidez de una imagen para optimizar la segmentación por SAM.
#Entrada:
# -imagen (np.ndarray): Imagen en formato BGR (OpenCV).    
#Devuelve:
# -np.ndarray: Imagen mejorada.
def mejorar_imagen_para_segmentacion(imagen: np.ndarray) -> np.ndarray:
    try:
        # Aumentar contraste y brillo
        imagen_contrastada = cv2.convertScaleAbs(imagen, alpha=1.3, beta=20)

        # Convertir a espacio LAB para mejorar iluminación
        lab = cv2.cvtColor(imagen_contrastada, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Equalización adaptativa en canal L (luminancia)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_mejorado = clahe.apply(l)

        # Reunir canales y volver a BGR
        lab_mejorado = cv2.merge((l_mejorado, a, b))
        imagen_mejorada = cv2.cvtColor(lab_mejorado, cv2.COLOR_LAB2BGR)
        
        return imagen_mejorada
    except Exception as e:
        return imagen