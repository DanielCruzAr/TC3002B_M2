# Clasificación multiclase de razas de perro
El archivo principal da 2 opciones:
1. Evaluar el modelo y mostrar la matriz de confusión
2. Hacer predicciones y mostrar los resultados como imágenes de perro con su respectiva raza
Para correrlo ejecutar `python main.py` y poner la opción como input.

**Nota**: Tendrás que cargar un modelo existente

Para entrenar una nueva versión del modelo ejecutar `python train_model.py`

Para cargar un modelo existente descárgalo de la siguiente carpeta: 
https://drive.google.com/drive/folders/16ucIzNjwrheYYTMhvggaL6Wbg5NPp-I_?usp=sharing

Una vez descargado ponlo en la carpeta de */models* y posteriormente en el archivo de `config.py` asegurate de poner el nombre del modelo descargado en la variable `MODELPATH`.

## Descripción
El objetivo de este proyecto es contruir un modelo que, dada una imagen de un perro, pueda decir a que raza pertenece.

## Datos
Los datos fueron recuperados de Kaggle en una competencia de identificación de razas de perro:
https://www.kaggle.com/c/dog-breed-identification/data [1]

### Set de datos de entrenamiento
https://drive.google.com/drive/folders/1LeCChExH5qRdSZfEmBIX5h7671LXL-ZB?usp=sharing

### Set de datos de prueba
https://drive.google.com/drive/folders/1u92xdDJYUEPuZNc1pH4E8Is1MhmZ4Q_S?usp=sharing

## Características
* Existen un total de 120 razas de perro, por lo cual hay 120 clases diferentes.
* Hay alrededor de 10,000 imágenes en el set de entrenamiento que cuentan con etiquetas de a qué raza pertenecen los perros.
* Hay alrededor de 10,000 imágenes en el set de prueba que no cuentan con etiquetas.
* El archivo `labels.csv` relaciona el id o nombre del archivo de imagen con su respectiva raza

## Datos de entrenamiento y validación
Como los datos de la carpeta *test* no cuentan con etiquetas se usará el mismo conjunto de datos de entrenamiento para validar el modelo. 

Estos datos estarán repartidos 80% para entrenamiento y 20% para validación del modelo.

## Preprocesamiento de datos
Se convertirán las imágenes a formato numérico en escala RGB y posteriormente se crearán batches a partir de los datos.

Inicialmente se pensaba convertir las imágenes a formato numérico leyendo las imágenes con la librería de Pillow para luego ajutar el tamaño a 224x224.
Una vez ajustado el tamaño se dividiría cada uno de los elemtentos del arreglo resultante entre 255 para que los números estuvieran en un rango del 0 al 1.
```
# Tamaño de imagen
IMG_SIZE = 224

# Función para preprocesamiento de imágenes
def process_image(image_path):
  # Leer el archivo de imagen
  image = Image.open(image_path)
  # Convertir la imagen a escala de grises si es necesario
  if image.mode != 'RGB':
      image = image.convert('RGB')
  # Cambiar el tamaño de imagen a (224, 224)
  image = image.resize((IMG_SIZE, IMG_SIZE))
  # Convertir la imagen a un array numpy y normalizar los valores
  image = np.array(image) / 255.0
  return image
```

Una vez teniendo la función para el procesamiento individual de imágenes se dividirían los datos en batches de tamaño 32 con el objetivo de alinearse a la arquitectura 
del modelo seleccionado. 
```
BATCH_SIZE = 32

# Regresa una tupla (image, label)
def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label

# Convertir los datos en batches
def create_data_batches(X, y=None, test_data=False):
    batches = []
    print("Creating data batches...")
    for i in range(0, len(X), BATCH_SIZE):
        print(f"Processing batch {i}...")
        if test_data:
            batch = X[i:i+BATCH_SIZE]
            labels = None
        else:
            batch = X[i:i+BATCH_SIZE]
            labels = y[i:i+BATCH_SIZE]
        images = []
        for image in batch:
            images.append(process_image(image))
        if test_data:
            batches.append(np.array(images))
        else:
            batches.append((np.array(images), np.array(labels)))
    return batches
```

Sin embargo, este método resulto ser ineficiente ya que el subconjunto de datos de entrenamiento tardo un total de 15 minutos en procesarse.

Debido a esto se decidió regresar a la idea original de utilizar la API de Tensorflow para preprocesar los datos, por lo cual a continuación se ofrece una explicación de las funciones utilizadas.

### Convertir imágenes a tensores
Un tensor es un arreglo multidimensional con una capacidad más alta que una matriz tradicional. 
* `tf.io.read_file()` lee el archivo en su formato original.
* `tf.image.decode_jpeg()` decodifica una imagen y la convierte a un tensor tipo int, `channels` indica los canales de color deseados, en este caso 3 son RGB en escala del 0 al 255.
* `tf.image.convert_image_dtype()` convierte una imagen a un tipo de dato especificado escalando sus valores si es necesario. Si el dato especifiado es de tipo flotante los valores de escalan del 0 al 1.
* `tf.image.resize()` redimensiona la imagen a los valores especificados.
### Dividir los datos en batches
* `tf.constant(X | y)` se usa para crear un tensor constante a partir de los nombres de archivo y las etiquetas para agrupar los datos. Si X es una lista de rutas de archivos de imágenes entonces, `tf.constant(X)` contendría una representación constante de las rutas de archivos de imágenes en forma de un tensor TensorFlow.
* `tf.data.Dataset.from_tensor_slices()` toma el tensor constante creado anteriormente y lo convierte en un conjunto de datos TensorFlow. Cada elemento de este conjunto de datos será un elemento de X. Si X es una lista de rutas de archivos de imágenes, entonces cada elemento del conjunto de datos será una ruta de archivo de imagen.
* `shuffle` se usa para revolver el dataset creado anteriormente con el objetivo de que el modelo entrene con los datos en orden aleatorio. Esto solo se usa para los datos de entrenamiento.
* `map()` se utiliza para aplicar una función a cada elemento del conjunto de datos. En este caso, la función que se aplica es *get_image_label*, la cual le asigna su etiqueta correspondiente a cada imagen. 
* `batch` se usa para agrupar los elementos de un conjunto de datos en lotes (batches) de un tamaño específico, en este caso 32. Si el número total de datos no es divisible de manera exacta por el tamaño del lote (BATCH_SIZE), TensorFlow ajustará automáticamente el último lote para incluir el número restante de datos. Esto significa que el último lote podría tener un tamaño menor que el tamaño del lote especificado.

Para más informaión consultar la documentación de la API de TensorFlow: https://www.tensorflow.org/api_docs

Se decidió dividir los datos en lotes o batches con el objetivo de que el modelo se adapte de manera más robusta a la distribución de los datos. Esto puede ayudar a prevenir el sobreajuste al presentar al modelo una variedad de ejemplos en cada paso de entrenamiento gracias a la aleatoriedad de los lotes.

Realizar este proceso significa tener que crear una función para desconvertir los datos en lotes para las predicciones de imágenes que haga el modelo porque los resultados deben poder ser interpretados por humanos.
## Selección del modelo
Para este proyecto se decidió utilizar un modelo basado en la arquitectura MobileNetV2, propuesta por Sandler et al. [2] la cual es una arquitectura de red neuronal convolucional (CNN) diseñada específicamente para aplicaciones de visión por computadora en dispositivos con recursos limitados, como dispositivos móviles y sistemas embebidos.
Algunas características de esta arquitectura son las siguientes:
* Bloques de construcción - Inverted Residuals with Linear Bottlenecks: MobileNetV2 utiliza una estructura de bloque de construcción llamada "Inverted Residuals with Linear Bottlenecks". Este bloque está diseñado para minimizar la cantidad de operaciones costosas en términos de computación, como las operaciones convolucionales estándar, mientras se mantiene la capacidad de representación de la red. Esto se logra mediante el uso de capas de convolución separables en profundidad (depthwise separable convolutions) y el uso de conexiones residuales.
* Capas Bottleneck: Cada bloque de construcción comienza con una capa de convolución 1x1 (conocida como la capa bottleneck) para reducir la dimensionalidad de los datos de entrada, seguida de una capa de convolución separable en profundidad para capturar patrones espaciales y una capa lineal para expandir la dimensionalidad nuevamente.
* Capa de Atención Espacial (Squeeze-and-Excitation): MobileNetV2 también implementa capas de atención espacial para ayudar a mejorar la capacidad del modelo para capturar características relevantes y suprimir el ruido. Esta capa, conocida como "Squeeze-and-Excitation", adapta dinámicamente la importancia de cada canal de características basándose en su relación espacial.
* Escala y Anchura de Reducción: MobileNetV2 introduce un hiperparámetro llamado "factor de escala" y un nuevo bloque de construcción llamado "bloque lineal" para permitir la modulación de la resolución de entrada y el tamaño del modelo. Esto proporciona un mayor control sobre la cantidad de recursos computacionales utilizados por el modelo y permite adaptarlo a diferentes restricciones de hardware.

El haber separado los datos en lotes ayuda a que el modelo reciba los datos con dimensionalidad reducida para la capa bottleneck.

Se utilizo un modelo secuencial preentrenado de tensorflow que ya divide las imágenes en pequeñas cajas con cada caja conteniendo un elemento de la imagen como una persona, un coche, o, en este caso, un perro. Este modelo recibe un tamaño de entrada ` [None, 224, 224, 3]` en donde None es el lote, 224 es la altura, 224 es la anchura y 3 es la cantidad de colores (rojo, verde y azul). Se agregó una capa Dense con 120 unidades para los datos de salida, las cuales representan las 120 clases existentes en este proyecto que son las 120 razas de perro diferentes. Se utiliza la función de activación `softmax` para obtener la probabilidad de cada clase.

Se utilizará la función `CategoricalCrossentropy()` para calcular la pérdida entre las etiquetas verdaderas y las predicciones del modelo. Se utilizará el optimizador *Adam* para ajustar los pesos de la red neuronal durante el proceso de entrenamiento. Finalmente, para evaluar el modelo se utilizará la métrica *accuracy* (exactitud) para evaluar el rendimiento del modelo. Esta métrica es la fracción de muestras clasificadas correctamente.

El modelo preentrenado fue recuperado de: https://www.kaggle.com/models/google/mobilenet-v2/tensorFlow2/130-224-classification/1 

## Resultados iniciales
Para la primera validación se entreno el modelo con solo 800 imágenes y se validó con 200, a su vez se ajustó con 10 épocas o iteraciones. Los resultados fueron los siguientes:
**Datos de entrenamiento:**
* Exactitud: 100.00%
* Pérdida: 0.04

**Datos de validación:**
* Exactitud: 66.50%
* Pérdida: 1.28

Una exactitud del 100% en en los datos de entrenamiento podría sugerir un posible sobreajuste, especialmente si la cantidad de datos de entrenamiento es relativamente pequeña. Pese a esto se obtuvo una exactitud del 66.5% en los datos de validación, lo cual sugiere que el modelo generaliza bien más allá de los datos de entrenamiento, pero aún hay margen para mejorar. A su vez, la pérdida del modelo en los datos de validación es considerable lo cual indica que el modelo está teniendo dificultades para hacer predicciones precisas en este conjunto de datos, ya que la pérdida es bastante alta en comparación con la pérdida en los datos de entrenamiento.

Considerando los resultados se concluye que el modelo es mejorable haciendo algunos ajustes como la cantidad de épocas a un número entre 10 y 100 y utilizando el total de imágenes.

## Referencias:
1. Will Cukierski. (2017). Dog Breed Identification. Kaggle. https://kaggle.com/competitions/dog-breed-identification
2. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," arXiv:1801.04381v4 [cs.CV], Mar. 2019.

