# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import pandas as pd
import numpy as np
import json
import gzip
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix

# Rutas
ruta_datos = "files/input"
ruta_modelo = "files/models/model.pkl.gz"
ruta_metricas = "files/output/metrics.json"

# Función para cargar datos desde archivos ZIP
def cargar_datos(ruta_archivo):
    return pd.read_csv(ruta_archivo, compression="zip")

# Cargar datos
train_df = cargar_datos(os.path.join(ruta_datos, "train_data.csv.zip"))
test_df = cargar_datos(os.path.join(ruta_datos, "test_data.csv.zip"))

# Procesamiento de datos
train_df.rename(columns={"default payment next month": "default"}, inplace=True)
test_df.rename(columns={"default payment next month": "default"}, inplace=True)
train_df.drop(columns=["ID"], inplace=True)
test_df.drop(columns=["ID"], inplace=True)
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)
train_df.loc[train_df["EDUCATION"] > 4, "EDUCATION"] = 4
test_df.loc[test_df["EDUCATION"] > 4, "EDUCATION"] = 4

X_train, y_train = train_df.drop(columns=["default"]), train_df["default"]
X_test, y_test = test_df.drop(columns=["default"]), test_df["default"]

# Transformaciones
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numerical_features = [col for col in X_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Definir pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=X_train.shape[1])),
        ("selector", SelectKBest(f_classif, k=15)),
        ("classifier", MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)),
    ]
)

# Validación cruzada
param_grid = {
    "classifier__hidden_layer_sizes": [(50,), (100,)],
    "classifier__alpha": [0.0001, 0.001],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring="balanced_accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Guardar modelo
with gzip.open(ruta_modelo, "wb") as f:
    joblib.dump(best_model, f)

# Predicciones
y_train_pred, y_test_pred = best_model.predict(X_train), best_model.predict(X_test)

# Métricas
metrics = [
    {
        "dataset": "train",
        "precision": classification_report(y_train, y_train_pred, output_dict=True)["1"]["precision"],
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": classification_report(y_train, y_train_pred, output_dict=True)["1"]["recall"],
        "f1_score": classification_report(y_train, y_train_pred, output_dict=True)["1"]["f1-score"],
    },
    {
        "dataset": "test",
        "precision": classification_report(y_test, y_test_pred, output_dict=True)["1"]["precision"],
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": classification_report(y_test, y_test_pred, output_dict=True)["1"]["recall"],
        "f1_score": classification_report(y_test, y_test_pred, output_dict=True)["1"]["f1-score"],
    },
]

# Matriz de confusión
train_cm, test_cm = confusion_matrix(y_train, y_train_pred), confusion_matrix(y_test, y_test_pred)

cm_metrics = [
    {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(train_cm[0, 0]), "predicted_1": int(train_cm[0, 1])},
        "true_1": {"predicted_0": int(train_cm[1, 0]), "predicted_1": int(train_cm[1, 1])},
    },
    {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(test_cm[0, 0]), "predicted_1": int(test_cm[0, 1])},
        "true_1": {"predicted_0": int(test_cm[1, 0]), "predicted_1": int(test_cm[1, 1])},
    },
]

# Guardar métricas
with open(ruta_metricas, "w") as f:
    json.dump(metrics + cm_metrics, f, indent=4)

