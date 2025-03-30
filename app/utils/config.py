from enum import Enum

# Formatos de imagen compatibles con Pillow
FORMATOS_COMPATIBLES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FORMATOS_TEXTO = ", ".join(ext.upper().replace(".", "") for ext in FORMATOS_COMPATIBLES)
MODELOS_SAM = {"vit_b": "facebook/sam-vit-base", "vit_l": "facebook/sam-vit-large", "vit_h": "facebook/sam-vit-huge"}
# Diccionario de modelos disponibles
MODEL_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}

class TipoInpainting(Enum):
    OpenCV = 0
    StableDiffusion = 1
    LaMa = 2