import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math
import os # Para crear directorios
# import subprocess # Descomentar si quieres usar la alternativa de subprocess para tensorflowjs_converter

# Funcion de normalización para los datos
def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)
  imagenes /= 255 # Aqui se pasa de 0-255 a 0-1
  return imagenes, etiquetas

def main():
    # Descargar set de datos de MNIST (Numeros escritos a mano, etiquetados)
    print("Descargando dataset MNIST...")
    datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)
    print("Dataset descargado.")

    # Obtener en variables separadas los datos de entrenamiento (60k) y pruebas (10k)
    datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

    # Normalizar los datos de entrenamiento y pruebas con la función que hicimos
    print("Normalizando datos...")
    datos_entrenamiento = datos_entrenamiento.map(normalizar)
    datos_pruebas = datos_pruebas.map(normalizar)
    print("Datos normalizados.")

    # Agregar a cache (usar memoria en lugar de disco, entrenamiento mas rapido)
    print("Agregando datos a caché...")
    datos_entrenamiento = datos_entrenamiento.cache()
    datos_pruebas = datos_pruebas.cache()
    print("Datos cacheados.")

    # (Opcional) Visualizar algunas imágenes del set de datos de entrenamiento
    # Este bloque es para inspección visual, puedes comentarlo si no lo necesitas cada vez.
    print("Mostrando ejemplos de imágenes del dataset...")
    clases = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.figure(figsize=(10,10))
    for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
      img_np = imagen.numpy().reshape((28,28)) # Convertir a numpy y redimensionar
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(img_np, cmap=plt.cm.binary)
      plt.xlabel(clases[etiqueta])
    plt.show()

    # Crear el modelo (Modelo denso, regular, sin redes convolucionales todavia)
    print("Creando el modelo...")
    modelo = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)), #1 = blanco y negro
        tf.keras.layers.Dense(units=50, activation='relu'),
        tf.keras.layers.Dense(units=50, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax') # Capa de salida para 10 clases con softmax
    ])
    print("Modelo creado.")

    # Compilar el modelo
    print("Compilando el modelo...")
    modelo.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    print("Modelo compilado.")

    # Obtener los números de datos de entrenamiento y pruebas
    num_datos_entrenamiento = metadatos.splits["train"].num_examples
    num_datos_pruebas = metadatos.splits["test"].num_examples # No se usa directamente para el entrenamiento pero es bueno tenerlo

    # Trabajar por lotes
    TAMANO_LOTE=32

    # Shuffle y repeat hacen que los datos esten mezclados de manera aleatoria
    # para que el entrenamiento no se aprenda las cosas en orden.
    # Preparar los datos para el entrenamiento.
    print("Preparando datos de entrenamiento y pruebas (batch, shuffle, repeat)...")
    datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_datos_entrenamiento).batch(TAMANO_LOTE)
    datos_pruebas = datos_pruebas.batch(TAMANO_LOTE) # No es necesario shuffle ni repeat para pruebas
    print("Datos listos para el entrenamiento.")

    # Realizar el entrenamiento
    print("Iniciando entrenamiento del modelo...")
    historial = modelo.fit(
        datos_entrenamiento,
        epochs=60, # Considera reducir las épocas para pruebas rápidas (ej. 5 o 10)
        steps_per_epoch=math.ceil(num_datos_entrenamiento/TAMANO_LOTE),
        # Es buena práctica añadir datos de validación para monitorear el sobreajuste:
        # validation_data=datos_pruebas,
        # validation_steps=math.ceil(num_datos_pruebas/TAMANO_LOTE)
    )
    print("Entrenamiento completado.")
    print("Historial de entrenamiento:", historial.history)

    # Guardar el modelo al explorador!
    nombre_archivo_modelo = 'numeros_regular.h5'
    modelo.save(nombre_archivo_modelo)
    print(f"Modelo guardado como '{nombre_archivo_modelo}'")

    # Convertirlo a tensorflow.js
    # Paso 1: Asegúrate de tener tensorflowjs instalado.
    # En la terminal de PyCharm (View > Tool Windows > Terminal), ejecuta:
    # pip install tensorflowjs
    # (Solo necesitas hacerlo una vez por entorno de PyCharm)

    # Paso 2: Crear la carpeta de salida si no existe
    carpeta_salida_tfjs = "carpeta_salida"
    os.makedirs(carpeta_salida_tfjs, exist_ok=True) # exist_ok=True evita error si ya existe
    print(f"Carpeta '{carpeta_salida_tfjs}' asegurada/creada.")

    # Paso 3: Ejecutar el convertidor.
    # Abre la terminal de PyCharm, navega al directorio donde está este script y ejecuta:
    print("\nPara convertir el modelo a TensorFlow.js, ejecuta el siguiente comando en tu terminal de PyCharm:")
    print(f"tensorflowjs_converter --input_format keras {nombre_archivo_modelo} {carpeta_salida_tfjs}")

    # Alternativa para ejecutar el comando desde Python (requiere que 'tensorflowjs_converter' esté en el PATH):
    # print("\nIntentando convertir el modelo a TensorFlow.js usando subprocess...")
    # try:
    #     comando = [
    #         "tensorflowjs_converter",
    #         "--input_format", "keras",
    #         nombre_archivo_modelo,
    #         carpeta_salida_tfjs
    #     ]
    #     resultado = subprocess.run(comando, check=True, capture_output=True, text=True)
    #     print("Conversión a TensorFlow.js completada exitosamente.")
    #     print("Salida del comando:", resultado.stdout)
    # except FileNotFoundError:
    #     print("Error: El comando 'tensorflowjs_converter' no se encontró. Asegúrate de que tensorflowjs esté instalado y en el PATH.")
    #     print("Instálalo con: pip install tensorflowjs")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error durante la conversión a TensorFlow.js: {e}")
    #     print("Error output:", e.stderr)
    # except Exception as e:
    #     print(f"Ocurrió un error inesperado durante la conversión: {e}")

if __name__ == "__main__":
    main()
