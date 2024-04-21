from data_preprocessing import get_test_and_valid_data, create_data_batches
from model import create_model, train_model
from model_evaluation import evaluate_model

def main():
    # Obtener datos de entrenamiento y validación
    filename = "data/labels.csv"
    train_path = "data/train/"
    n_images = 1000
    
    X_train, X_val, y_train, y_val = get_test_and_valid_data(filename, train_path, n_images=n_images)

    # Crear batches de datos de entrenamiento y validación
    train_data = create_data_batches(X_train, y_train)
    validation_data = create_data_batches(X_val, y_val, valid_data=True)

    # Definir modelo
    input_shape = train_data.element_spec[0].shape
    output_shape = 120 # Número de razas de perros
    model_url = "https://kaggle.com/models/google/mobilenet-v2/TensorFlow2/130-224-classification/1"
    model = create_model(input_shape, output_shape, model_url)

    # Entrenar el modelo
    num_epochs = 10
    fitted_model = train_model(num_epochs, model, train_data, validation_data)

    # Evaluar el modelo
    evaluate_model(fitted_model.history)
    
if __name__ == "__main__":
    main()