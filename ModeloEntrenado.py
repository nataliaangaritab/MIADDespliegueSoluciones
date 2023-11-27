#!/usr/bin/env python
# coding: utf-8

# In[4]:


import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.ensemble import RandomForestClassifier


# Deshabilitar todos los warnings
warnings.filterwarnings("ignore")
    
import pandas as pd
from sklearn.model_selection import train_test_split
# Se carga el archivo de datos ya pre-procesado
df = pd.read_csv("data/datos_VF.csv")
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
# Se define los parámetros del modelo
n_estimators = 250 
max_depth = 5
max_features = 10
# Se crea el modelo con los parámetros definidos y se entrena
rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
rf.fit(X_train, y_train)
# Exportar modelo a archivo binario .pkl
joblib.dump(rf, os.getcwd()+'/Modelo_energy.pkl', compress=3)

