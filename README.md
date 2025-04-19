# **<u>HUD Remover X10</u>**

### Introducción

Este proyecto nace para aplicar los conocimientos adquiridos en el curso de "**Desarrollador 10x**" impartido por el **Instituto de Inteligencia Artificial (IIA)**.

El proyecto consiste en una aplicación con interfaz Gradio en la que podemos subir nuestras imagenes, segmentarlas y elegir las capas que queremos eliminar de la imagen usando inpainting.

De los conceptos vistos en el curso, hemos aplicado:

- **SAM (Segment Anything Model):** mediante el cual, segmentaremos la imagen en diferentes capas. Se ha implementado el uso de 3 modelos a cada cual mas preciso (sacrificando velocidad por precisión).
  
- **Pipeline**: para hacer uso de Stable Diffusion desde Huggingface.
  
- **Gradio**: para la propia interfaz de la aplicación.
  
- **Stable Diffussion**: para realizar inpainting avanzado.
  

La necesidad de hacer esta aplicación surge ya que, en mis ratos libres hago fotografia virtual que consiste en hacer capturas de videojuegos y, a veces cuando juego a algun juego que no tiene Modo Foto, las interfaces de los juegos se mantienen presentes y "estropean" la imagen. Entonces aprovechando el poder de la IA, he decidido ver si podia desarrollar algo que me ayudara a eliminar el grueso de esas interfaces.

Esto es una primera version funcional, por lo que no es perfecta, pero es efectiva. Ademas, la he probado con fotografia real eliminando algun elemento, y no lo hace nada mal.

### Tipos de Inpainting

Se han implementado 3 tipos de inpainting usando diferentes herramientas:

- **OpenCV**: es una librería gratuita y de código abierto que sirve para que los ordenadores puedan ver, entender y trabajar con imágenes y vídeos. Con ella hacemos inpainting basico (funciona muy bien en elementos pequeños).
  
- **Lama**: es un modelo de IA diseñado para rellenar zonas faltantes o eliminar objetos de imágenes de forma realista. Utiliza redes neuronales profundas entrenadas con millones de ejemplos para reconstruir el contenido perdido manteniendo la coherencia visual.
  
- **Stable Diffusion**: es un modelo de IA que genera imágenes a partir de texto de forma rápida y con alta calidad.
  

### Futuras mejoras:

Se ha pensado en futuras mejoras que, por tiempo limitado, no se han podido realizar:

1. **Segmentación por puntos**: consiste en marcar en la imagen puntos y despues segmentar la imagen en base a esos puntos.
  
2. **Segmentación por cajas**: permitir al usuario generar cajas con las zonas donde se desea buscar la segmentación.
  
3. **Segmentación por pincel**: permitir al usuario "pintar" sobre la zona que desea buscar segmentación.
  
4. **Upscaling**: mejora de resolución de la imagen.

## Demo video:
[Video demo de HUD Remover X](https://youtu.be/ssSD1Vz9OP8?si=xLClDEr4YQ0B7vzI)

[![Video demo de HUD Remover X](https://i.ytimg.com/vi/ssSD1Vz9OP8/maxresdefault.jpg)]([https://www.youtube.com/watch?v=FEa2diI2qgA](https://youtu.be/ssSD1Vz9OP8?si=xLClDEr4YQ0B7vzI))

## Como instalar y ejecutar la aplicación:

A continuacion se detalla la manera de descargar y ejecutar la aplicación. Se hace el tutorial ya que, para poder usar Lama y Stable Diffusion, es necesario tener una tarjeta grafica dedicada (con al menos **6 Gb de VRAM**) y hay que ejecutar ciertos comandos para instalar dependencias que no se han podido incluir en el fichero de **requirements.txt** tambien indicar que es necesario Python 3.11 para ejecutar la aplicación.

- Descargar el proyecto manualmente desde este repositorio o por consola
  
- Por comodidad y no ensuciar el equipo es necesario crear un entorno virtual para ejecutar la aplicación y descargar ahi las dependencias
  
  ```
  C:\Users\[tuusuario]\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv        
  ```
  
- Activar el entorno virtual
  
  ```.venv\Scripts\activate
  .venv\Scripts\activate
  ```
  
- Istalar el fichero requirements.txt
  
  ```pip
  pip install -r requirements.txt
  ```
  

Si disponemos de tarjeta gráfica dedicada, tendremos que lanzar este comando ya que es probable que el fichero requirements nos instale una version de torch para CPU y no la version CUDA por lo que habra comprobar usando este comando la version que tenemos:

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

Lanzar la aplicación:

```bash
python -m app.main
```

