from abc import ABC, abstractmethod
from typing import List, Dict, Type
from PIL import Image
from app.sam.sam_loader import MascaraSegmentada
from app.utils.config import TipoInpainting

#Clase base abstracta que define una interfaz común para los motores de inpainting.
#Esta clase debe ser heredada por cada implementación concreta (OpenCV, LaMa, Stable Diffusion, etc.).
#Permite desacoplar la lógica de interfaz del motor usado, facilitando el uso modular y escalable.
class InpaintingBase(ABC):


    # Diccionario interno que mapea tipos a clases concretas
    _registry: Dict[TipoInpainting, Type["InpaintingBase"]] = {}

    # Tipo asociado al motor (se define en la subclase)
    tipo: TipoInpainting

    #Decorador para registrar automáticamente una subclase en la factoría.
    #Recibe:
    #   -tipo (TipoInpainting): Tipo asociado a la subclase.
    @classmethod
    def register(cls, tipo: TipoInpainting):        
        
        def decorator(subclass):
            cls._registry[tipo] = subclass
            return subclass
        return decorator

    #Método factoría que crea una instancia del inpainter correspondiente.
    #Recibe:
    #   -tipo (TipoInpainting): Tipo de motor deseado.
    #Devuelve:
    #   -InpaintingBase: Instancia de la subclase registrada.
    @classmethod
    def crear(cls, tipo: TipoInpainting) -> "InpaintingBase":
       
        if tipo not in cls._registry:
            raise ValueError(f"No hay un inpainter registrado para {tipo}")
        return cls._registry[tipo]()  # Instancia la clase correspondiente

    #Método abstracto para cargar o inicializar el modelo de inpainting.
    @abstractmethod
    def cargar_modelo(self):
        pass

    #Método abstracto que recibe una imagen y una máscara y devuelve una nueva imagen con los objetos de la máscara eliminados.
    #Recibe:
        # -imagen (Image.Image): Imagen de entrada (PIL).
        # -mascaras (np.ndarray): Máscara binaria en formato NumPy, donde 1 = área a eliminar.
    #Devuelve:
        # -Image.Image: Imagen de salida con el área eliminada/inpainted.        
    @abstractmethod
    def eliminar_objetos(self, imagen: Image.Image, mascaras: List[MascaraSegmentada]) -> Image.Image:        
        pass
