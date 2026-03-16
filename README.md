# Detección de Movimiento en Videos largos con Inteligencia Artificial

Herramienta en Python para analizar videos y extraer automáticamente imágenes relevantes utilizando detección de movimiento y modelos de inteligencia artificial.

El programa permite revisar grabaciones largas de forma rápida y guardar solo los frames donde ocurre algo importante, como movimiento, presencia de personas o detección de rostros.

El proyecto utiliza OpenCV para el procesamiento de video y OpenVINO para ejecutar modelos optimizados de detección.

---

## Características

- Extracción automática de imágenes desde video
- Detección de movimiento usando MOG2
- Detección de personas mediante modelo OpenVINO
- Detección de rostros mediante modelo OpenVINO
- Interfaz gráfica desarrollada con PyQt5
- Posibilidad de elegir intervalo de frames
- Redimensionado automático de imágenes
- Marca de tiempo en las capturas (timestamp)
- Sistema opcional de salto de tiempo después de cada captura

---

## Tecnologías utilizadas

- Python
- OpenCV
- OpenVINO
- NumPy
- PyQt5

---

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/luisdl-dev/video-mov-deteccion.git
cd video-frame-extractor-ai
