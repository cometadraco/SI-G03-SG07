import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

#Cargar el conjunto de datos
df = pd.read_csv('SI_L07_SVM_DATASET.csv', encoding='latin1')

#Preprocesamiento de datos
df_model = df.drop(['Rank', 'Country'], axis=1)
le = LabelEncoder()
df_model['Continent_num'] = le.fit_transform(df_model['Continent'])

#Visualización inicial de datos (opcional)
sns.pairplot(df_model, hue='Continent')
plt.show()

#División en datos de entrenamiento y prueba (70% - 30%)
X = df_model[['Overall Life', 'Male Life', 'Female Life']]
y = df_model['Continent_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Entrenar modelo SVM con kernel RBF
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

#Evaluación del modelo
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.4f}")
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

#Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

#Visualización con gráficos de barras promedio por continente

# Esperanza de vida general promedio
plt.figure(figsize=(12,5))
promedios_general = df_model.groupby('Continent')['Overall Life'].mean().sort_values(ascending=False)
sns.barplot(x=promedios_general.index, y=promedios_general.values)
plt.title('Esperanza de Vida General Promedio por Continente')
plt.ylabel('Esperanza de Vida Promedio')
plt.xlabel('Continente')
plt.show()

# Esperanza de vida masculina promedio
plt.figure(figsize=(12,5))
promedios_male = df_model.groupby('Continent')['Male Life'].mean().sort_values(ascending=False)
sns.barplot(x=promedios_male.index, y=promedios_male.values)
plt.title('Esperanza de Vida Masculina Promedio por Continente')
plt.ylabel('Esperanza de Vida Promedio')
plt.xlabel('Continente')
plt.show()

# Esperanza de vida femenina promedio
plt.figure(figsize=(12,5))
promedios_female = df_model.groupby('Continent')['Female Life'].mean().sort_values(ascending=False)
sns.barplot(x=promedios_female.index, y=promedios_female.values)
plt.title('Esperanza de Vida Femenina Promedio por Continente')
plt.ylabel('Esperanza de Vida Promedio')
plt.xlabel('Continente')
plt.show()

#Predicción con datos nuevos
nuevos_datos = pd.DataFrame([[75, 72, 78]], columns=['Overall Life', 'Male Life', 'Female Life'])
prediccion_num = svm_model.predict(nuevos_datos)[0]
prediccion_continente = le.inverse_transform([prediccion_num])[0]
print(f"Predicción de continente para datos {nuevos_datos.values.tolist()}: {prediccion_continente}")
