import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras

def create_model(input_shape, output_shape, url):
    # Declarar las capas del modelo
    model = keras.Sequential([
        hub.KerasLayer(url),
        keras.layers.Dense(output_shape, activation='softmax')
    ])
    
    # Compilar el modelo
    model.compile(optimizer='adam',
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])
    
    # Construir el modelo
    model.build(input_shape)
    
    return model

def train_model(num_epochs, model, train_data, validation_data):
    # Entrenar el modelo
    model.fit(x=train_data,
            epochs=num_epochs,
            validation_data=validation_data,
            validation_freq=1)
    
    return model

def load_model(modelpath):
    return keras.models.load_model(modelpath,
                                    custom_objects={'KerasLayer': hub.KerasLayer})