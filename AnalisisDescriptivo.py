#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
# Desactivar todas las advertencias temporalmente
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("data/datosv1.csv")
df.head()


# ### Dimensiones de la data

# In[3]:


df.shape


# In[4]:


# Calcular la cantidad total de valores nulos en el DataFrame
total_nulos = df.isna().sum().sum()

# Calcular el porcentaje total de valores nulos
porcentaje_nulos = (total_nulos / (df.shape[0] * df.shape[1])) * 100

print(f"Porcentaje total de valores nulos en el DataFrame: {porcentaje_nulos:.2f}%")


# ### Limpieza y estandarización preliminar de datos

# In[5]:


# Las siguientes columnas se eliminan por no tener información relevante para el análisis
# Unnamed: 0, se elimina porque corresponde al número de la observación y la indexación cumple esa función
# ID_UC, se elimina la unidad de consumo porque no es relevante
# DATA_REFERENCIA, se elimina porque no es de interés eliminar la data como una serie de tiempo
# ID_OCORRENCIA, se elimina el ID de ocurrencia porque no es relevante

columnas_a_borrar = ["Unnamed: 0", "ID_UC", "DATA_REFERENCIA", "ID_OCORRENCIA"]
df = df.drop(columnas_a_borrar, axis=1)
df.head()


# In[6]:


pd.set_option('display.max_rows', None)
# Contar la cantidad de nulos por columna
cantidad_nulos = df.isnull().sum()
# Calcular el porcentaje de nulos sobre el total de cada columna
porcentaje_nulos = (cantidad_nulos / len(df)) * 100
# Crear un nuevo DataFrame para mostrar los resultados
resultados = pd.DataFrame({'Cantidad de Nulos': cantidad_nulos, 'Porcentaje de Nulos (%)': porcentaje_nulos})
resultados.sort_values(by='Porcentaje de Nulos (%)', ascending=False)


# In[7]:


resultados_ordenados = resultados.sort_values(by='Porcentaje de Nulos (%)', ascending=False)
variables_con_nulos_mayor_a_40 = resultados_ordenados[resultados_ordenados['Porcentaje de Nulos (%)'] > 40]


# In[8]:


# Obtiene la lista de nombres de variables resultante
lista_variables = variables_con_nulos_mayor_a_40.index.tolist()
print(lista_variables)


# In[9]:


df = df.drop(lista_variables, axis=1)
df.head()


# Las variables ENERGIA_A_INCREMENTAR y ENERGIA_A_RECUPERAR para el conjunto de datos abordado, se excluirán debiado a que tienen una dependencia directa  con la variable resultado.

# In[10]:


df = df.drop(['ENERGIA_A_INCREMENTAR', 'ENERGIA_A_RECUPERAR'], axis=1)
df.head()

# Eliminar fechas
df = df.drop(['D100', 'D224'], axis=1)
df.head()


# Los valores de la columna RESULTADO corresponden a 10 cuando la inspección a la instalación se determina normal y 7 cuando la inspección arroja un fraude, que son las instalaciones que nos interesa verificar.

# In[11]:


column_means = df.mean().round(0)
df = df.fillna(column_means)
df.head()


# ### Gráficas descriptivas

# In[12]:


# Contar los valores únicos en la columna 'Resultado'
conteo_resultados = df['RESULTADO'].value_counts()

# Crear un gráfico de torta con los valores agregados a las etiquetas
plt.figure(figsize=(5, 5))
# plt.pie(conteo_resultados, labels=[f'{label} ({count})' for label, count in conteo_resultados.iteritems()], autopct='%1.1f%%', startangle=90)
plt.title('Distribución de fraudes')
plt.axis('equal')  # Para que el gráfico sea un círculo en lugar de una elipse

# Mostrar el gráfico de torta
plt.show()


# In[13]:


# Crear un histograma de la columna 'N294'
plt.hist(df['N294'], bins=20, edgecolor='k')
plt.xlabel('Cantidad')
plt.ylabel('Frecuencia')
plt.title('Número de revisiones ejecutadas último año')
plt.grid(True)

# Mostrar el histograma
plt.show()


# In[14]:


# Filtrar los valores menores a 100,000 en la columna 'N479'
valores_filtrados = df[df['N479'] < 10000]['N479']

# Crear un histograma para los valores filtrados
plt.figure(figsize=(8, 6))
plt.hist(valores_filtrados, bins=20, edgecolor='k')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Suma de los consumos leídos de los últimos 3 meses (kWh) (Valores < 100,000)')
plt.grid(True)

# Mostrar el histograma
plt.show()


# In[15]:


# Selecciona las columnas desde N445 hasta N478
columnas_a_promediar = df.loc[:, 'N445':'N478']

# Calcula el promedio de estas columnas
promedio_columnas = columnas_a_promediar.mean(axis=1)
promedio_columnas.describe()


# ### Elaboración de un dataframe para principales variables relevantes

# In[16]:


dfmedidas = pd.DataFrame({'Promedio de facturación mensual': promedio_columnas})

dfmedidas = dfmedidas.merge(df[['N294']], left_index=True, right_index=True)
dfmedidas = dfmedidas.rename(columns={'N294': 'Revisiones ejecutadas último año'})

dfmedidas = dfmedidas.merge(df[['RESULTADO']], left_index=True, right_index=True)
dfmedidas = dfmedidas.rename(columns={'RESULTADO': 'RESULTADO'})

dfmedidas = dfmedidas.merge(df[['N479']], left_index=True, right_index=True)
dfmedidas = dfmedidas.rename(columns={'N479': 'Suma de los consumos leídos de los últimos 3 meses (kWh)'})

df['N479']

dfmedidas.head()


# In[17]:


dfmedidas.describe()


# In[18]:


dfmedidas["Revisiones ejecutadas último año"].value_counts()
# Obtener los valores y sus conteos
conteos_revisiones = dfmedidas["Revisiones ejecutadas último año"].value_counts().reset_index()

# Renombrar las columnas
conteos_revisiones.columns = ['Revisiones', 'Cantidad']

# Crear un nuevo DataFrame
df_resultado = pd.DataFrame(conteos_revisiones)
df_resultado

