# 1. Importar librerias necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Carga de datos
data = pd.read_csv('dataset_regresion_logistica.csv')

# Explorar el conjunto de datos
print(data.head())
print(data.info())
print(data.describe())

# Preparacion de los datos
# Separar las variables independientes (X) y la dependiente (y)
X = data.drop('target', axis=1)
y = data['target']

# Dividir el dataset en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos (opcional pero recomendable para regresion logistica)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del modelo
# Crear el modelo de Regresion Logistica
logistic_model = LogisticRegression()

# Entrenar el modelo
logistic_model.fit(X_train_scaled, y_train)

# Realizar predicciones con el conjunto de prueba
y_pred = logistic_model.predict(X_test_scaled)

# Evaluacion del modelo
# Crear la matriz de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusion
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Imprimir el reporte de clasificacion
print(classification_report(y_test, y_pred))

# Imprimir la exactitud del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud del modelo: {accuracy * 100:.2f}%')

