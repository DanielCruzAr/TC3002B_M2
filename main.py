from model import load_model
from data_preprocessing import get_data, get_test_and_valid_data, create_data_batches
from model_evaluation import evaluate_model, get_confusion_matrix, plot_confusion_matrix

def main():
    # Cargar el modelo
    model = load_model()
    
    # Obtener datos de entrenamiento y validación
    filename = "data/labels.csv"
    train_path = "data/train/"
    filenames, bool_labels, unique_breeds = get_data(filename, train_path)
    
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
    
if __name__ == "__main__":
    main()