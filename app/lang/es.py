# Diccionario de cadenas de texto en español
from app.utils.config import FORMATOS_TEXTO

text_gradio_ui = {
    "titulo_app": "🖼️ HUD Remover X10",
    "subir_imagen": "Sube tu imagen",
    "propiedades_imagen": "Propiedades de la imagen",
    "sin_imagen": "No se ha cargado ninguna imagen.",
    "propiedades_formato": "📐 Dimensiones: {width} x {height}\n🎨 Canales: {channels}\n🧬 Modo: {mode}",
    "error_cargar_imagen": "❌ Error al cargar la imagen: {error}",
    "archivo_nombre": "🗂️ Nombre del archivo: {nombre}",
    "imagen_dimensiones": "📐 Dimensiones: {ancho} x {alto}",
    "imagen_modo_color": "🎨 Modo de color: {modo}",
    "imagen_formato": "🧾 Formato de archivo: {formato}",
    "imagen_canales": "📊 Canales: {canales}",
    "exif_titulo": "\n📷 EXIF:",
    "exif_no_disponible": "\n📷 EXIF: No disponible",
    "exif_error": "\n📷 EXIF: Error al leer ({error})",
    "formatos_compatibles": "📝 Formatos compatibles: {formatos}",    
    "preview_imagen": "📷 Previsualización de la imagen:", 
}

text_image_info = {
    "archivo_nombre": "🗂️ Nombre del archivo: {nombre}",
    "imagen_dimensiones": "📐 Dimensiones: {ancho} x {alto}",
    "imagen_modo_color": "🎨 Modo de color: {modo}",
    "imagen_formato": "🧾 Formato de archivo: {formato}",
    "imagen_canales": "📊 Canales: {canales}",
    "exif_titulo": "\n📷 EXIF:",
    "exif_no_disponible": "\n📷 EXIF: No disponible",
    "exif_error": "\n📷 EXIF: Error al leer ({error})",
}

text_general ={
        "formato_no_soportado":"⚠️ El formato de la imagen no está soportado. Usa uno de los siguientes: {formatos}"
}