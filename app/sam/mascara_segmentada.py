from typing import Tuple
from PIL import Image
import numpy as np
from dataclasses import dataclass

# Clase para encapsular información de cada máscara
@dataclass
class MascaraSegmentada:
    binaria: np.ndarray                      # Máscara binaria (0 o 1)
    color: Tuple[int, int, int]              # Color RGB asignado
    miniatura: Image.Image                   # Imagen de vista previa coloreada
    sam_original: dict                       # Objeto original de SAM con metadatos
    id:int = -1                              # ID de la máscara (opcional, para referencia)