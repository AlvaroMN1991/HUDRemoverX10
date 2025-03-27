# Formatos de imagen compatibles con Pillow
FORMATOS_COMPATIBLES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FORMATOS_TEXTO = ", ".join(ext.upper().replace(".", "") for ext in FORMATOS_COMPATIBLES)