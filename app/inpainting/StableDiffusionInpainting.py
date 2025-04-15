from app.inpainting.InpaintingBase import InpaintingBase
from app.sam.mascara_segmentada import MascaraSegmentada
from app.utils.config import TipoInpainting
from PIL import Image
from typing import List, Optional
import torch
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import cv2

@InpaintingBase.register(TipoInpainting.StableDiffusion) #Esto es para que luego desde la interfaz, se pueda crear una clase de manera implicita
class StableDiffusionInpainting(InpaintingBase):
    
    CROP_MARGIN = 64
    
    #Implementación del motor de inpainting usando Stable Diffusion.
    #Usa la pipeline de Hugging Face 'runwayml/stable-diffusion-inpainting'.
    def __init__(self) -> None:
        self.pipe: Optional[StableDiffusionInpaintPipeline] = None

    #Carga el modelo de Hugging Face para realizar inpainting.
    def cargar_modelo(self):  
        try:#
            #self.pipe = AutoPipelineForInpainting.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16)
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
            self.pipe.enable_model_cpu_offload()            
            self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images)) # 🚫 Desactivar filtro NSFW. Saltaba mucho y devolvia un cuadrado negro xD 
        except Exception as e:
            print(f"Error cargando el modelo: {e}")  

    @staticmethod
    def crop_around_mask(image_np, mask_np, margin=64):
        try:
            ys, xs = np.where(mask_np > 0)
            if len(xs) == 0 or len(ys) == 0:
                return image_np, mask_np, (0, 0)

            x1 = max(xs.min() - margin, 0)
            x2 = min(xs.max() + margin, image_np.shape[1])
            y1 = max(ys.min() - margin, 0)
            y2 = min(ys.max() + margin, image_np.shape[0])

            cropped_img = image_np[y1:y2, x1:x2]
            cropped_mask = mask_np[y1:y2, x1:x2]
            return cropped_img, cropped_mask, (x1, y1)
        except Exception as e:
            print(f"❌ Error en crop_around_mask: {e}")
            return image_np, mask_np, (0, 0)
    
    #Pega un parche (patch) sobre una imagen base (base_img) de forma segura, evitando errores de desbordamiento de índices.
    #Entrada:
    #   -base_img (np.ndarray): Imagen base donde se quiere pegar el parche.
    #   -patch (np.ndarray): Imagen/parche que se quiere pegar.
    #   -offset_x (int): Coordenada X donde empieza el pegado.
    #   -offset_y (int): Coordenada Y donde empieza el pegado.
    #Retorno:
    #   -np.ndarray: Imagen con el parche pegado correctamente.
    @staticmethod
    def safe_paste_patch(base_img, patch, offset_x, offset_y):    
        try:
            h_base, w_base = base_img.shape[:2]
            h_patch, w_patch = patch.shape[:2]

            # Asegurar que los offsets no son negativos
            offset_x = max(0, offset_x)
            offset_y = max(0, offset_y)

            # Limitar el tamaño del parche a lo que cabe en la imagen base
            max_h = min(h_patch, h_base - offset_y)
            max_w = min(w_patch, w_base - offset_x)

            # Pegar el parche recortado
            base_img[offset_y:offset_y + max_h, offset_x:offset_x + max_w] = patch[:max_h, :max_w]
            return base_img
        except Exception as e:
            print(f"❌ Error al pegar el parche: {e}")
            return base_img
    
    def eliminar_objetos(self, imagen: Image.Image, mascaras: List[MascaraSegmentada]) -> Image.Image:
        try:
            if self.pipe is None:
                raise RuntimeError("❌ El modelo no se ha cargado.")

            # Convertir imagen a array
            image_np = np.array(imagen.convert("RGB"))

            # 🟣 Combinar todas las máscaras en una sola binaria
            mask_np = np.zeros((imagen.height, imagen.width), dtype=np.uint8)
            for m in mascaras:
                mask_np |= m.binaria.astype(np.uint8)

            # 🟡 Suavizar bordes con blur para evitar cortes bruscos
            mask_np = cv2.GaussianBlur(mask_np, (15, 15), 0).astype(np.uint8)

            # 🔵 Opcional: dilatar máscara para cubrir un poco más
            kernel = np.ones((5, 5), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=1).astype(np.uint8)

            # 🔶 Recorte inteligente con margen
            cropped_img, cropped_mask, (offset_x, offset_y) = self.crop_around_mask(image_np, mask_np, margin=self.CROP_MARGIN)

            # Asegurar que el tamaño del recorte sea múltiplo de 64 (requisito SD)
            h_crop, w_crop = cropped_img.shape[:2]
            new_w = (w_crop // 64) * 64
            new_h = (h_crop // 64) * 64
            cropped_img = cropped_img[:new_h, :new_w]
            cropped_mask = cropped_mask[:new_h, :new_w]

            # Convertir a PIL
            cropped_img_pil = Image.fromarray(cropped_img)
            cropped_mask_pil = Image.fromarray((cropped_mask * 255).astype(np.uint8)).convert("L")

            # 🧠 Prompt + negative_prompt para mejorar calidad del relleno
            prompt = ""
            negative_prompt = ""

            # 🧪 Llamar al modelo
            result_pil = self.pipe(prompt="", image=cropped_img_pil, mask_image=cropped_mask_pil).images[0] # type: ignore
            # Convertir resultado a array
            result_np = np.array(result_pil)

            # Asegurar que resultado tenga tamaño exacto al esperado
            result_h, result_w = result_np.shape[:2]
            expected_h, expected_w = cropped_img.shape[:2]
            if (result_h, result_w) != (expected_h, expected_w):
                result_np = cv2.resize(result_np, (expected_w, expected_h), interpolation=cv2.INTER_LANCZOS4)

            # 🧩 Reinsertar el parche generado en la imagen original
            final_img = image_np.copy()
            final_img = self.safe_paste_patch(final_img, result_np, offset_x, offset_y)

            return Image.fromarray(final_img)

        except Exception as e:
            print(f"❌ Error en SD eliminar_objetos: {e}")
            return imagen

    
    
