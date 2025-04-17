import subprocess
import zipfile
import os
import gdown #type:ignore
from pathlib import Path
from typing import Tuple

# Rutas base
LAMA_REPO_URL = "https://github.com/advimman/lama.git"
LAMA_DIR = Path("app/lama/lamarepo")
CONFIG_PATH = "app/lama/lama_config.yaml"
RUTA_PESO_MODELO="app/lama/big-lama/models/best.ckpt"
#LaMa Models Repo --> https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips?usp=drive_link

# URLs oficiales
CKPT_DRIVE_ID = "11RbsVSav3O-fReBsPHBE1nn8kcFIMnKp"
CONFIG_URL = "https://raw.githubusercontent.com/advimman/lama/main/configs/prediction/default.yaml"


def clonar_repo_lama():
    if not LAMA_DIR.exists():
        print("[LaMa] Clonando repositorio...")
        subprocess.run(["git", "clone", LAMA_REPO_URL, str(LAMA_DIR)], check=True)
    else:
        print("[LaMa] Repositorio ya existe.")

def get_lama_paths() -> Tuple[str, str]:
    """
    Asegura que el repositorio y los pesos están listos. Devuelve:
    - Ruta al config.yaml
    - Ruta al best.ckpt
    """
    clonar_repo_lama()

    return str(CONFIG_PATH), str(RUTA_PESO_MODELO)
