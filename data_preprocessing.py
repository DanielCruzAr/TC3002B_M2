import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Tamaño de imagen
IMG_SIZE = 224

# Tamaño de batch
BATCH_SIZE = 32

def get_data(filename, train_path):
    labels_csv = pd.read_csv(filename)
    filenames = [train_path + fname + ".jpg" for fname in labels_csv["id"]]
    labels = labels_csv["breed"].to_numpy()
    unique_breeds = np.unique(labels)
    bool_labels = [label == unique_breeds for label in labels]
    return filenames, bool_labels

def get_test_and_valid_data(filename, train_path, n_images=1000):
    X, y = get_data(filename, train_path)

    X_train, X_val, y_train, y_val = train_test_split(X[:n_images],
                                                  y[:n_images],
                                                  test_size=0.2,
                                                  random_state=42)
    
    return X_train, X_val, y_train, y_val

# Función para preprocesamiento de imágenes
def process_image(image_path):
  # Leer el archivo de imagen
  image = tf.io.read_file(image_path)
  # Convertir la imagen a formato numérico con RGB
  image = tf.image.decode_jpeg(image, channels=3)
  # Convertir los valores RGB de 0-255 a 0-1
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Cambiar el tamaño de imagen a (224, 224)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

  return image

# Regresa una tupla (image, label)
def get_image_label(image_path, label):
  image = process_image(image_path)
  return image, label

# Convertir los datos en batches
def create_data_batches(X, y=None, valid_data=False):
  if valid_data:
    print("Creando datos de validación")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),  # imagenes
                                               tf.constant(y))) # etiquetas
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    print("Creando datos de entrenamiento")
    # Convertir imagenes y etiquetas en numeros
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),
                                               tf.constant(y)))
    #  Revolver las imagenes y etiquetas
    data = data.shuffle(buffer_size=len(X))

    # Crear tuplas (image, label)
    data = data.map(get_image_label)

    # Convertir a batches
    data_batch = data.batch(BATCH_SIZE)
  return data_batch