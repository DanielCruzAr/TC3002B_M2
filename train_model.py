from data_preprocessing import get_data, get_test_and_valid_data, create_data_batches
from model import create_model, train_model
from model_evaluation import plot_model_history
from config import MODELPATH

def main():
    # Obtener datos de entrenamiento y validación
    filename = "data/labels.csv"
    train_path = "data/train/"
    filenames, bool_labels, unique_breeds = get_data(filename, train_path)
    
    X_train, X_val, y_train, y_val = get_test_and_valid_data(filenames, bool_labels)

    # Crear batches de datos de entrenamiento y validación
    train_data = create_data_batches(X_train, y_train)
    validation_data = create_data_batches(X_val, y_val, valid_data=True)

    # Definir modelo
    input_shape = train_data.element_spec[0].shape
    output_shape = len(unique_breeds)
    model = create_model(input_shape, output_shape)

    # Entrenar el modelo
    fitted_model = train_model(model, train_data, validation_data)
    
    # Evaluar el modelo
    plot_model_history(fitted_model.history.history)
    
    # Guardar el modelo entrenado
    fitted_model.save(MODELPATH)
    
if __name__ == "__main__":
    main()