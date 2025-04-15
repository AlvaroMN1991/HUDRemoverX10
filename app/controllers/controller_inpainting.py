from PIL import Image
from app.globals.estado_global import EstadoGlobal
from app.inpainting.InpaintingBase import InpaintingBase
from app.utils.config import TipoInpainting

# Ejecuta la eliminación de objetos usando el motor seleccionado
def ejecutar_inpainting( metodo: int, indices: list[str], estado: EstadoGlobal) -> Image.Image:
    try:
        if not indices or not estado.mascaras_memoria:
            return Image.new("RGB", (512, 512), (255, 0, 0))

        seleccionados = list(map(int, indices))
        mascaras = [estado.mascaras_memoria[i] for i in seleccionados]

        inpainter = InpaintingBase.crear(TipoInpainting(metodo))
        inpainter.cargar_modelo()

        if estado.imagen_pil is None:
            return Image.new("RGB", (512, 512), (255, 0, 0))  # O cualquier mensaje visual de error

        # Luego sí llamas
        resultado = inpainter.eliminar_objetos(estado.imagen_pil, mascaras)

        return resultado
    
    except Exception as e:
        print(f"❌ Error en ejecutar_inpainting: {e}")
        return Image.new("RGB", (512, 512), (255, 0, 0))
