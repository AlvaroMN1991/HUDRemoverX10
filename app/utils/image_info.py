from PIL import Image, ExifTags
from typing import Union
from app.lang.es import text_gradio_ui
import os

def obtener_propiedades_imagen(image: Image.Image, nombre_archivo: Union[str, None] = None) -> str:
    propiedades = []

    if nombre_archivo:
        propiedades.append(text_gradio_ui["archivo_nombre"].format(nombre=os.path.basename(nombre_archivo)))

    ancho, alto = image.size
    propiedades.append(text_gradio_ui["imagen_dimensiones"].format(ancho=ancho, alto=alto))
    propiedades.append(text_gradio_ui["imagen_modo_color"].format(modo=image.mode))
    propiedades.append(text_gradio_ui["imagen_formato"].format(formato=image.format or "Desconocido"))
    propiedades.append(text_gradio_ui["imagen_canales"].format(canales=len(image.getbands())))

    try:
        exif = image.getexif()
        if exif:
            propiedades.append(text_gradio_ui["exif_titulo"])
            for tag_id, valor in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                propiedades.append(f"   - {tag}: {valor}")
        else:
            propiedades.append(text_gradio_ui["exif_no_disponible"])
    except Exception as e:
        propiedades.append(text_gradio_ui["exif_error"].format(error=e))

    return "\n".join(propiedades)
