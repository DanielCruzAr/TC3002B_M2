import matplotlib.pyplot as plt

def get_results(history):
    # Get the accuracy and loss values from the history object
    accuracy = history['accuracy'][-1]
    loss = history['loss'][-1]

    # Get the validation accuracy and loss values from the history object
    val_accuracy = history['val_accuracy'][-1]
    val_loss = history['val_loss'][-1]

    return accuracy, loss, val_accuracy, val_loss

def evaluate_model(history):
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