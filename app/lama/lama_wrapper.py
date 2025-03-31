import sys
from omegaconf import OmegaConf
from pathlib import Path
from app.utils.tools import get_device_id
from app.lama.lama_downloader import get_lama_paths
        
_model_cache = {} #type:ignore

#Descarga LaMa si es necesario, ajusta el sys.path e importa el modelo.
#Devuelve el modelo y utilidades necesarias.

def load_lama_model():
    try:
        if _model_cache.get("modelo"):
            return _model_cache["modelo"]

        # Obtener rutas
        config_path, checkpoint_path = get_lama_paths()

        # Añadir repo al path si hace falta
        lama_path = Path("app/lama/lamarepo")
        if str(lama_path) not in sys.path:
            sys.path.insert(0, str(lama_path))

        # Importaciones diferidas
        #Ignoramos los errores por que deberian cargarse en el primer uso del modelo
        from app.lama.lamarepo.saicinpainting.training.trainers import load_checkpoint #type:ignore
        from app.lama.lamarepo.saicinpainting.evaluation.data import pad_img_to_modulo #type:ignore
        from app.lama.lamarepo.saicinpainting.evaluation.utils import move_to_device #type:ignore

        # Cargar el config.yaml como objeto OmegaConf
        config = OmegaConf.load(config_path)
        # Establecer el checkpoint manualmente (si hace falta)
        config.model.path = checkpoint_path        
        config.training_model.predict_only = True

        device = get_device_id()
        model = load_checkpoint(config, checkpoint_path, strict=False, map_location=device)
        model.freeze()
        model.to(device)

        _model_cache["modelo"] = (model, device, pad_img_to_modulo, move_to_device)
        return _model_cache["modelo"]
    except Exception as e:
        print(f"Error en el módulo lama_wrapper.py:{e}" )
        return []