# HUD Remover X10

Meetodo de instalación de la app.

1. Crear un entorno virtual con la version 3.11.3 (importante) para no "ensuciar" el equipo:
```bash
C:\Users\[tuusuario]\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
```

2. Activar el entorno virtual:
```bash
.venv\Scripts\activate
```

3. Instalar las dependencias:
```bash
pip install -r requirements.txt
```
Es probable que torch no instale la version CUDA por lo que habra comprobar usando este comando la version que tenemos:
```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```
Si pone False habra que desinstalar usando el comando:
```bash
pip uninstall torch torchvision torchaudio -y
```
Y lanzar este comando de manera manual para instalar: 
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
4. Lanzar la aplicación:
```bash
python app/main.py
```

