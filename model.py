import tensorflow as tf
import tensorflow_hub as hub

def create_model(input_shape, output_shape, url):
    # Declarar las capas del modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: hub.KerasLayer(url)(x)),
        tf.keras.layers.Dense(output_shape, activation='softmax')
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
    history = model.fit(x=train_data,
                        epochs=num_epochs,
                        validation_data=validation_data,
                        validation_freq=1)
    
    return history