# segmentadores/segmentador_base.py
from abc import ABC, abstractmethod
from app.utils.config import TipoSegmentacion
from app.sam.sam_loader import MascaraSegmentada
import numpy as np, random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Type, Any


@dataclass
class Segmentacion:
    tipo: TipoSegmentacion
    puntos: List[Tuple[int, int]]  # Puntos marcados por el usuario
    etiquetas: List[int]           # 1=foreground, 0=background

    def es_valida(self) -> bool:
        return len(self.puntos) > 0

#Interfaz base para todos los métodos de segmentación con SAM.
class SegmentadorBase(ABC):

    def __init__(self, predictor):
        self.predictor = predictor

    # Diccionario interno que mapea tipos a clases concretas
    _registry: Dict[TipoSegmentacion, Type["SegmentadorBase"]] = {}

    # Tipo asociado al motor (se define en la subclase)
    tipo: TipoSegmentacion

    #Decorador para registrar automáticamente una subclase en la factoría.
    #Recibe:
    #   -tipo (TipoSegmentacion): Tipo asociado a la subclase.
    @classmethod
    def register(cls, tipo: TipoSegmentacion):        
        
        def decorator(subclass):
            cls._registry[tipo] = subclass
            return subclass
        return decorator

    #Método factoría que crea una instancia del segmentador correspondiente.
    #Recibe:
    #   -tipo (TipoSegmentacion): Tipo de motor deseado.
    #Devuelve:
    #   -SegmentadorBase: Instancia de la subclase registrada.
    @classmethod
    def crear(cls, tipo: TipoSegmentacion, predictor: Any) -> "SegmentadorBase":
       
        if tipo not in cls._registry:
            raise ValueError(f"No hay un inpainter registrado para {tipo}")
        return cls._registry[tipo](predictor)  # Instancia la clase correspondiente

    #Aplica segmentación sobre la imagen usando la anotación proporcionada.
    @abstractmethod
    def segmentar(self, imagen: np.ndarray, segmentacion: Segmentacion) -> List[MascaraSegmentada]:
        pass

    # 🎨 Función para generar un color aleatorio
    @staticmethod
    def generar_color_aleatorio(self) -> tuple[int, int, int]:
        try:
            r = random.randint(50, 200)
            g = random.randint(50, 200)
            b = random.randint(50, 200)
            return (r, g, b)
        except Exception as e:
            return (0, 0, 0)

    # 🖼️ Función para crear una miniatura coloreada desde una máscara
    @staticmethod
    def generar_miniatura_color(self, binaria: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        miniatura = np.zeros((*binaria.shape, 3), dtype=np.uint8)
        for i in range(3):
            miniatura[:, :, i] = binaria * color[i]
        return miniatura