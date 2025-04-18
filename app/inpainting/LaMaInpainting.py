from typing import List
from PIL import Image
import numpy as np
import torch
import cv2
from app.sam.mascara_segmentada import MascaraSegmentada
from app.inpainting.InpaintingBase import InpaintingBase, TipoInpainting
from app.lama.lama_wrapper import load_lama_model

#Inpainter basado en LaMa (Look At My Assumptions).
#Usa la versión big-lama entrenada por SAIC-AI.
@InpaintingBase.register(TipoInpainting.LaMa) #Esto es para que luego desde la interfaz, se pueda crear una clase de manera implicita
class LaMaInpainting(InpaintingBase):

    tipo = TipoInpainting.LaMa # Para la clase base

    def __init__(self):
        super().__init__()  

        self.model = None
        self.device = None
        self.pad_img_to_modulo = None
        self.move_to_device = None

    def cargar_modelo(self):
        try:
            self.model, self.device, self.pad_img_to_modulo, self.move_to_device = load_lama_model()
        except Exception as e:
            print(f"Error cargando el modelo: {e}")

    def crop_around_mask(self, image, mask, margin=64):
        try:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                return image, mask, (0, 0)

            x1, x2 = max(xs.min() - margin, 0), min(xs.max() + margin, image.shape[1])
            y1, y2 = max(ys.min() - margin, 0), min(ys.max() + margin, image.shape[0])

            cropped_img = image[y1:y2, x1:x2]
            cropped_mask = mask[y1:y2, x1:x2]
            return cropped_img, cropped_mask, (x1, y1)
        except Exception as e:
            print(f"[LamaInpainter] Error durante crop_around_mask: {e}")
            return [], [], []

    def safe_paste_patch(self, base_img, patch, offset_x, offset_y):
        try:
            h_base, w_base = base_img.shape[:2]
            h_patch, w_patch = patch.shape[:2]

            offset_x = max(0, offset_x)
            offset_y = max(0, offset_y)

            max_h = min(h_patch, h_base - offset_y)
            max_w = min(w_patch, w_base - offset_x)

            base_img[offset_y:offset_y + max_h, offset_x:offset_x + max_w] = patch[:max_h, :max_w]
            return base_img
        except Exception as e:
            print(f"[LamaInpainter] Error durante safe_paste_patch: {e}")
            return Image.new("RGB", (512, 512), (255, 0, 0))

    #Mejora una máscara binaria aplicando dilatación y desenfoque gaussiano.
    #Entrada:
    # -mascara (np.ndarray): Máscara en formato 2D (H, W) con valores 0-255 o 0-1.
    # -kernel_size (int): Tamaño del kernel para dilatación y blur.
    # -sigma (int): Sigma para el desenfoque gaussiano.
    # -dilatacion_iter (int): Número de iteraciones de dilatación.        
    #Devuelve:
    # -np.ndarray: Máscara mejorada (valores 0 o 255).
    def mejorar_mascara(self, mascara: np.ndarray, kernel_size: int = 5, sigma: int = 3, dilatacion_iter: int = 1) -> np.ndarray:
        
        try:
            # Asegurar que está en uint8 y en rango 0-255
            if mascara.max() <= 1:
                mascara = (mascara * 255).astype(np.uint8)
            else:
                mascara = mascara.astype(np.uint8)

            # Dilatar la máscara (ensanchar bordes)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mascara_dilatada = cv2.dilate(mascara, kernel, iterations=dilatacion_iter)

            # Suavizar los bordes con desenfoque
            mascara_suavizada = cv2.GaussianBlur(mascara_dilatada, (kernel_size, kernel_size), sigmaX=sigma)

            # Binarizar nuevamente (evita sombras grises)
            return (mascara_suavizada > 10).astype(np.uint8) * 255
        except Exception as e:
            print(f"[LamaInpainter] Error durante inpainting: {e}")
            return mascara
        
    #Aplica un suavizado en los bordes del parche usando la máscara como alfa de mezcla.
    def suavizar_transicion(self, parche: np.ndarray, original: np.ndarray, mascara: np.ndarray, feather_size: int = 15) -> np.ndarray:
        try:
            # Asegurar que la máscara tiene un canal
            if mascara.ndim == 3:
                mascara = mascara[:, :, 0]

            # Crear máscara de mezcla difuminada
            alpha = cv2.GaussianBlur(mascara, (feather_size, feather_size), sigmaX=feather_size//2)
            alpha = alpha.astype(np.float32) / 255.0
            alpha = np.clip(alpha, 0, 1)

            # Aplicar mezcla entre parche y original
            parche_fusionado = parche.astype(np.float32) * alpha[..., None] + original.astype(np.float32) * (1 - alpha[..., None])
            return parche_fusionado.astype(np.uint8)
        except Exception as e:
            return parche
        
     #Elimina objetos en la imagen usando LaMa. Combina todas las máscaras y realiza inpainting con LaMa.
    
    def eliminar_objetos(self, imagen: Image.Image, mascaras: List[MascaraSegmentada], prompt_sd: str="", negative_prompt_sd: str="") -> Image.Image:
    
        try:
            # Convertimos la imagen a NumPy (RGB)
            imagen_np = np.array(imagen.convert("RGB"))  # (H, W, 3)
            altura, anchura = imagen_np.shape[:2]

            # --- Crear máscara combinada ---
            mascara_total: np.ndarray = np.zeros((altura, anchura), dtype=np.uint8)
            for m in mascaras:
                binaria = m.binaria
                if binaria.max() <= 1:
                    binaria = (binaria * 255).astype(np.uint8)
                else:
                    binaria = binaria.astype(np.uint8)

                if binaria.shape != mascara_total.shape:
                    binaria = cv2.resize(binaria, (anchura, altura), interpolation=cv2.INTER_NEAREST)

                binaria = np.ascontiguousarray(binaria, dtype=np.uint8)
                mascara_total = cv2.bitwise_or(mascara_total, binaria)

            # --- Recorte inteligente alrededor de la máscara ---
            # 🔥 Aplicamos mejora visual a la máscara antes de recortar
            mascara_mejorada = self.mejorar_mascara(mascara_total, kernel_size=5, sigma=3, dilatacion_iter=1)
            cropped_img, cropped_mask, (offset_x, offset_y) = self.crop_around_mask(imagen_np, mascara_mejorada, margin=64)

            # --- Preprocesamiento: CHW + Padding ---
            imagen_chw = cropped_img.transpose(2, 0, 1)  # (3, H, W)
            imagen_chw = np.ascontiguousarray(imagen_chw, dtype=np.uint8)

            # ⚠️ Calcular padding manualmente (no podemos modificar pad_img_to_modulo)
            _, h, w = imagen_chw.shape
            out_h = (h + 7) // 8 * 8
            out_w = (w + 7) // 8 * 8
            pad_h = out_h - h
            pad_w = out_w - w

            # Aplicamos padding a la imagen con pad_img_to_modulo
            imagen_padded = self.pad_img_to_modulo(imagen_chw, mod=8)

            # --- Preparar máscara en formato (1, H, W) ---
            if cropped_mask.ndim == 2:
                mascara_chw = cropped_mask[np.newaxis, :, :]
            elif cropped_mask.ndim == 3 and cropped_mask.shape[0] < 10:
                mascara_chw = cropped_mask.transpose(2, 0, 1)
            else:
                raise ValueError(f"Forma inesperada de máscara: {cropped_mask.shape}")

            mascara_chw = np.ascontiguousarray(mascara_chw, dtype=np.uint8)

            # Aplicamos el mismo padding que a la imagen
            mascara_padded = np.pad(mascara_chw,((0, 0), (0, pad_h), (0, pad_w)),mode="symmetric")

            # --- Crear batch para LaMa ---
            batch = {
                'image': torch.from_numpy(imagen_padded / 255.0).unsqueeze(0).float().to(self.device),  # [1, 3, H, W]
                'mask': torch.from_numpy(mascara_padded / 255.0).unsqueeze(0).float().to(self.device)   # [1, 1, H, W]
            }

            # --- Inferencia con LaMa ---
            with torch.no_grad():
                salida = self.model(batch)

            resultado = salida['inpainted'][0].cpu().permute(1, 2, 0).numpy()  # (H, W, C)
            resultado = (resultado * 255).astype(np.uint8)

            # --- Recorte padding post-inpainting ---
            if pad_h > 0 or pad_w > 0:
                resultado = resultado[:-pad_h or None, :-pad_w or None, :]

            parche_suavizado = self.suavizar_transicion(resultado, cropped_img, cropped_mask)
            final_img = self.safe_paste_patch(imagen_np.copy(), parche_suavizado, offset_x, offset_y)

            # --- Pegamos el resultado de vuelta sobre la imagen original ---
            final_img = self.safe_paste_patch(imagen_np.copy(), resultado, offset_x, offset_y)
            return Image.fromarray(final_img)

        except Exception as e:
            print(f"[LamaInpainter] Error durante inpainting: {e}")
            return Image.new("RGB", (512, 512), (255, 0, 0))
