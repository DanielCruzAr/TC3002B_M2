# Clasificación multiclase de razas de perro
Ejecutar el comando `python main.py`

## Descripción
El objetivo de este proyecto es contruir un modelo que, dada una imagen de un perro, pueda decir a que raza pertenece.

## Datos
Los datos fueron recuperados de Kaggle en una competencia de identificación de razas de perro:
https://www.kaggle.com/c/dog-breed-identification/data

### Set de datos de entrenamiento
https://drive.google.com/drive/folders/1LeCChExH5qRdSZfEmBIX5h7671LXL-ZB?usp=sharing

### Set de datos de prueba
https://drive.google.com/drive/folders/1-UGH4BtS4LyJO2V1k9KwiC1o8gx0tYfT?usp=sharing

## Características
* Existen un total de 120 razas de perro, por lo cual hay 120 clases diferentes.
* Hay alrededor de 10,000 imágenes en el set de entrenamiento que cuentan con etiquetas de a qué raza pertenecen los perros.
* Hay alrededor de 10,000 imágenes en el set de prueba que no cuentan con etiquetas.
* El archivo `labels.csv` relaciona el id o nombre del archivo de imagen con su respectiva raza
