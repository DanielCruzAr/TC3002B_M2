# Clasificación multiclase de razas de perro
Ejecutar el comando `python main.py`

## Descripción
El objetivo de este proyecto es contruir un modelo que, dada una imagen de un perro, pueda decir a que raza pertenece.

## Datos
Los datos fueron recuperados de Kaggle en una competencia de identificación de razas de perro:
https://www.kaggle.com/c/dog-breed-identification/data

### Set de datos de entrenamiento
https://drive.google.com/drive/folders/1LeCChExH5qRdSZfEmBIX5h7671LXL-ZB?usp=sharing

### Set de datos de prueba
https://drive.google.com/drive/folders/1-UGH4BtS4LyJO2V1k9KwiC1o8gx0tYfT?usp=sharing

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
### Dividir los datos en batches

## Selección del modelo

## Resultados iniciales
