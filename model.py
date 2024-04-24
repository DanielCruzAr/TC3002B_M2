import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
from config import MODELPATH, NUM_EPOCHS, VALIDATION_FREQ

def create_model(input_shape, output_shape, url):
    # Declarar las capas del modelo
    model = keras.Sequential([
        hub.KerasLayer(url),
        # TODO: Poner capas dropout o weight decay
        keras.layers.Dense(output_shape, activation='softmax')
    ])
    
    # Compilar el modelo
    model.compile(optimizer='adam',
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])
    
    # Construir el modelo
    model.build(input_shape)
    
    return model

def train_model(model, train_data, validation_data):
    # Entrenar el modelo
    model.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=validation_data,
            validation_freq=VALIDATION_FREQ)
    
    return model

def load_model(modelpath=MODELPATH):
    return keras.models.load_model(modelpath,
                                    custom_objects={'KerasLayer': hub.KerasLayer})