from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.losses import CategoricalCrossentropy
from config import MODELPATH, NUM_EPOCHS, VALIDATION_FREQ

def create_model(input_shape, output_shape):
    base_model = MobileNetV2(
        alpha=1.0,
        include_top=False,
        input_shape=(input_shape[1], input_shape[2], input_shape[3]), # (224, 224, 3)
        weights='imagenet',
        pooling=None,
    )
    base_model.trainable = False
    
    # Declarar las capas del modelo
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(output_shape, activation='softmax'))
    
    # Compilar el modelo
    model.compile(optimizer='adam',
            loss=CategoricalCrossentropy(),
            metrics=['accuracy'])
    
    return model

def train_model(model, train_data, validation_data):
    # Entrenar el modelo
    model.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=validation_data,
            validation_freq=VALIDATION_FREQ)
    
    return model

def load_model(modelpath=MODELPATH):
    return models.load_model(modelpath)