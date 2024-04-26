from model import load_model
from data_preprocessing import (
    get_train_data, 
    get_test_data, 
    get_test_and_valid_data, 
    create_data_batches,
    get_prediction_label,
    unbatch_data
)
from model_evaluation import evaluate_model, get_confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

def rate_model(model):
    # Obtener datos de entrenamiento y validación
    filename = "data/labels.csv"
    train_path = "data/train/"
    filenames, bool_labels, unique_breeds = get_train_data(filename, train_path)
    
    _, X_val, _, y_val = get_test_and_valid_data(filenames, bool_labels)

    # Crear batches de datos de entrenamiento y validación
    validation_data = create_data_batches(X_val, y_val, valid_data=True)

    # Evaluar el modelo
    results = evaluate_model(model, validation_data)
    print("Exactitud: {:.2f}%".format(results[1] * 100))
    print("Pérdida: {:.2f}".format(results[0]))
    
    # Obtener y graficar la matriz de confusión
    cm = get_confusion_matrix(model, validation_data, y_val)
    plot_confusion_matrix(cm, unique_breeds)
    
def make_predictions(model):
    # Obtener datos de prueba
    test_path = "data/test/"
    filenames, unique_breeds = get_test_data(test_path)
    test_data = create_data_batches(filenames, test_data=True)
    
    # Hacer predicciones
    predictions = model.predict(test_data)
    
    # Obtener etiquetas de las predicciones
    pred_labels = [get_prediction_label(p, unique_breeds) for p in predictions]
    
    # Desconvertir las imagenes en lotes
    images = unbatch_data(test_data)
    
    # Graficar las predicciones
    plt.figure(figsize=(10,10))
    for i in range(len(images)):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(pred_labels[i])
    plt.show()

def main():
    # Cargar el modelo
    model = load_model()
    
    option = int(input("[1] para evaluar el modelo, [2] para hacer predicciones: "))
    if option == 1:
        rate_model(model)
    elif option == 2:
        make_predictions(model)
    else:
        print("Opción inválida")
    
if __name__ == "__main__":
    main()