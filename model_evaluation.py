import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.losses import CategoricalCrossentropy

def get_results(history):
    # Get the accuracy and loss values from the history object
    accuracy = history['accuracy'][-1]
    loss = history['loss'][-1]

    # Get the validation accuracy and loss values from the history object
    val_accuracy = history['val_accuracy'][-1]
    val_loss = history['val_loss'][-1]

    return accuracy, loss, val_accuracy, val_loss

def evaluate_model(model, validation_data):
    # Compilar el modelo
    model.compile(optimizer='adam',
            loss=CategoricalCrossentropy(),
            metrics=['accuracy'])
    
    # Evaluate the model
    results = model.evaluate(validation_data)
    return results

def plot_model_history(history):
    # Get the accuracy and loss values from the history object
    accuracy = history['accuracy']
    loss = history['loss']

    # Get the validation accuracy and loss values from the history object
    val_accuracy = history['val_accuracy']
    val_loss = history['val_loss']
    
    print("Datos de entrenamiento:")
    print("Exactitud: {:.2f}%".format(accuracy[-1] * 100))
    print("Pérdida: {:.2f}".format(loss[-1]))
    print("\nDatos de validación:")
    print("Exactitud: {:.2f}%".format(val_accuracy[-1] * 100))
    print("Pérdida: {:.2f}".format(val_loss[-1]))

    # Plot the accuracy curves
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot the loss curves
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def get_confusion_matrix(model, validation_data, y_val):
    predictions = model.predict(validation_data, verbose=1)
    y_preds = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_val, axis=1)
    cm = confusion_matrix(y_true, y_preds)
    return cm

def plot_confusion_matrix(cm, class_labels):
    for i in range(len(class_labels)):
        print(f"{i}: {class_labels[i]}")
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()