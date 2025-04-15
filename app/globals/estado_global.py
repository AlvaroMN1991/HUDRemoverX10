from typing import Optional, List, Tuple
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from app.sam.mascara_segmentada import MascaraSegmentada

#Clase para mantener el estado de la aplicación de forma centralizada.
#Evita el uso de variables globales sueltas y facilita pruebas y modularidad.
@dataclass
class EstadoGlobal:
                
        imagen_base_np: Optional[np.ndarray] = None
        imagen_pil: Optional[Image.Image] = None
        puntos_usuario: List[Tuple[int, int]] = field(default_factory=list)
        mascaras_memoria: List[MascaraSegmentada] = field(default_factory=list)