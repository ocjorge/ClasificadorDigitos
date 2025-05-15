# Clasificador de Dígitos MNIST con TensorFlow y TensorFlow.js

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js)
[![Dataset](https://img.shields.io/badge/Dataset-MNIST-lightgrey)](http://yann.lecun.com/exdb/mnist/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Opcional: Build Status (si usas GitHub Actions, etc.) -->
<!-- [![Build Status](https://github.com/TU_NOMBRE_DE_USUARIO/NOMBRE_DEL_REPOSITORIO/actions/workflows/python-app.yml/badge.svg)](https://github.com/TU_NOMBRE_DE_USUARIO/NOMBRE_DEL_REPOSITORIO/actions/workflows/python-app.yml) -->

Este proyecto contiene un script de Python que entrena un modelo de red neuronal simple para clasificar dígitos escritos a mano del popular dataset MNIST. El modelo se entrena utilizando TensorFlow y Keras. Además, se proporcionan instrucciones para convertir el modelo entrenado al formato TensorFlow.js para su uso en aplicaciones web.
Este colab forma parte del video de Redes Neuronales Convolucionales del canal de Youtube "Ringa Tech"
https://youtu.be/eGDSlW93Bng

## Características

*   Descarga y preprocesa el dataset MNIST.
*   Define y entrena un modelo secuencial de Keras.
*   Normaliza los datos de imagen para un mejor rendimiento.
*   Utiliza caché para acelerar el acceso a los datos durante el entrenamiento.
*   Visualiza ejemplos de imágenes del dataset (opcional).
*   Guarda el modelo entrenado en formato HDF5 (`.h5`).
*   Incluye instrucciones para convertir el modelo `.h5` a formato TensorFlow.js.

## Requisitos Previos

*   Python 3.7 o superior
*   pip (gestor de paquetes de Python)

## Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/TU_NOMBRE_DE_USUARIO/NOMBRE_DEL_REPOSITORIO.git
    cd NOMBRE_DEL_REPOSITORIO
    ```

2.  **(Recomendado) Crea y activa un entorno virtual:**
    ```bash
    python -m venv venv
    # En Windows:
    # .\venv\Scripts\activate
    # En macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Instala las dependencias:**
    Crea un archivo `requirements.txt` con el siguiente contenido:
    ```txt
    tensorflow
    tensorflow-datasets
    matplotlib
    tensorflowjs
    ```
    Luego, instálalas:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

### 1. Entrenar el Modelo

Ejecuta el script principal para descargar los datos, entrenar el modelo y guardarlo:

```bash
python entrenar_mnist.py
