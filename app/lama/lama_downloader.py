import subprocess
import zipfile
import os
import gdown #type:ignore
from pathlib import Path
from typing import Tuple

# Rutas base
LAMA_REPO_URL = "https://github.com/advimman/lama.git"
LAMA_DIR = Path("app/lama/lamarepo")
BIG_LAMA_DIR = LAMA_DIR / "big-lama"
MODELS_DIR = BIG_LAMA_DIR / "models"
CKPT_PATH = MODELS_DIR / "best.ckpt"
CONFIG_PATH = "app/lama/lama_config.yaml"

#LaMa Models Repo --> https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips?usp=drive_link

# URLs oficiales
CKPT_DRIVE_ID = "1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips"
CONFIG_URL = "https://raw.githubusercontent.com/advimman/lama/main/configs/prediction/default.yaml"


def clonar_repo_lama():
    if not LAMA_DIR.exists():
        print("[LaMa] Clonando repositorio...")
        subprocess.run(["git", "clone", LAMA_REPO_URL, str(LAMA_DIR)], check=True)
    else:
        print("[LaMa] Repositorio ya existe.")


def descargar_pesos_big_lama():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if not CKPT_PATH.exists():
            print("[LaMa] Descargando pesos de Big-LaMa...")
            gdown.download(id=CKPT_DRIVE_ID, output=str(CKPT_PATH), quiet=False)
            print("[LaMa] Descomprimiendo pesos...")
            with zipfile.ZipFile(CKPT_PATH, 'r') as zip_ref:
                zip_ref.extractall(MODELS_DIR)

            os.remove(CKPT_PATH)
        else:
            print("[LaMa] Pesos ya están descargados.")
    except Exception as e:
        print(f"[LaMa] ❌ Error descargando los pesos: {e}")
        print("[LaMa] Sugerencia: descárgalos manualmente desde:")
        print("  https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips?usp=drive_link")
        print("Y colócalos en: external/lama/big-lama/models/best.ckpt")


def get_lama_paths() -> Tuple[str, str]:
    """
    Asegura que el repositorio y los pesos están listos. Devuelve:
    - Ruta al config.yaml
    - Ruta al best.ckpt
    """
    clonar_repo_lama()
    descargar_pesos_big_lama()

    return str(CONFIG_PATH), str(CKPT_PATH)
