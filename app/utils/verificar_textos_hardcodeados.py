import os
import re

EXCLUIR_RUTAS = [
    "venv", ".venv", "__pycache__", "checkpoints", "models"
]

EXTENSIONES_VALIDAS = (".py",)

def es_excluido(ruta):
    return any(excluido in ruta for excluido in EXCLUIR_RUTAS)

def buscar_cadenas_hardcodeadas(base_dir="app"):
    patron = re.compile(r'(?<!TEXTOS\[)\"(.*?)\"|\'(.*?)\'')  # cadenas que no están dentro de TEXTOS[]
    resultados = []

    for dirpath, _, archivos in os.walk(base_dir):
        if es_excluido(dirpath):
            continue

        for archivo in archivos:
            if archivo.endswith(EXTENSIONES_VALIDAS):
                ruta_completa = os.path.join(dirpath, archivo)
                with open(ruta_completa, encoding="utf-8") as f:
                    for numero_linea, linea in enumerate(f, start=1):
                        if '#' in linea:
                            linea = linea.split('#')[0]  # ignoramos comentarios

                        matches = patron.findall(linea)
                        for match in matches:
                            texto = match[0] if match[0] else match[1]
                            if len(texto.strip()) > 1 and not texto.startswith("http"):
                                resultados.append((ruta_completa, numero_linea, texto.strip()))

    return resultados

if __name__ == "__main__":
    cadenas = buscar_cadenas_hardcodeadas()
    if cadenas:
        print("⚠️  Cadenas de texto hardcodeadas encontradas:\n")
        for archivo, linea, texto in cadenas:
            print(f"{archivo}:{linea} → \"{texto}\"")
    else:
        print("✅ No se encontraron cadenas hardcodeadas.")
