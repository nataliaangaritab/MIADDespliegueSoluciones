#!/usr/bin/env python
# coding: utf-8

"""
Created on Sat Nov 25 2023

@author: Marcell Piraquive
"""

import warnings
# Deshabilitar todos los warnings
warnings.filterwarnings("ignore")
    
import pandas as pd
from sklearn.model_selection import train_test_split
# Se carga el archivo de datos ya pre-procesado
df = pd.read_csv("./datos_VF.csv")
columnas_a_convertir = ['ID_UC_READING_CYCLE', 'ID_MUNICIPALITY', 'ID_REGION', 'UC_COL_ID_05']
df[columnas_a_convertir] = df[columnas_a_convertir].astype(object)
# Cambiar valores específicos en la columna 'variable'
valores_a_cambiar = {10: 0, 7: 1}  # Diccionario de valores a cambiar {valor_original: valor_nuevo}
df['RESULTADO'] = df['RESULTADO'].replace(valores_a_cambiar)

# Se borrra la etiqueta
X = df.drop("Unnamed: 0",axis=1)
X = X.drop("RESULTADO",axis=1)
# Guarda las etiquetas en un objeto de la serie Pandas
y = df['RESULTADO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Se importa MLFlow para registrar los experimentos, el clasificador de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Se registra el experimento
experiment = mlflow.set_experiment("sklearn-diab")

# Se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno 
# por defecto y guarda las características del experimento y las métricas definidas. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Se define los parámetros del modelo
    n_estimators = 250 
    max_depth = 5
    max_features = 10
    # Se crea el modelo con los parámetros definidos y se entrena
    rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    rf.fit(X_train, y_train)
    # Realice predicciones de prueba
    predictions = rf.predict(X_test)
  
    # Se registra los parámetros
    mlflow.log_param("num_trees", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("max_feat", max_features)
  
    # Se registra el modelo
    mlflow.sklearn.log_model(rf, "random-forest-model")
  
    # Se crea y registra las métricas de interés
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    print(mse)
    
    accuracy  = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy ", accuracy )
    print(accuracy)
    

