#Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Cargar el dataset
df = pd.read_csv('calidadvino.csv', sep=';')

#Exploración inicial de datos
print("Primeras filas del dataset:")
print(df.head())
print("\nInformación general:")
print(df.info())
print("\nDescripción estadística:")
print(df.describe())

#Visualizar la distribución de la variable calidad
plt.figure(figsize=(8,4))
sns.countplot(x='quality', data=df)
plt.title('Distribución de la Calidad del Vino')
plt.show()

#Matriz de correlación para ver relaciones entre variables
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()

#Separar variables independientes (X) y variable objetivo (y)
X = df.drop(columns=['quality'])
y = df['quality']

#Dividir el dataset en entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrenar un modelo de Árbol de Decisión con profundidad inicial 3
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X_train, y_train)

#Realizar predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# 10. Evaluar rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo (profundidad=3): {accuracy:.3f}")
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

#Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print("\nImportancia de las variables:")
print(importances)

plt.figure(figsize=(8,5))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Importancia de las características')
plt.show()

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())],
          filled=True, rounded=True, fontsize=12)
plt.title('Árbol de Decisión - profundidad 3')
plt.show()

max_depths = range(1, 15)
accuracies = []
for depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_depth = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred_depth)
    accuracies.append(acc)

plt.figure(figsize=(8,5))
plt.plot(max_depths, accuracies, marker='o')
plt.title('Precisión vs Profundidad del Árbol')
plt.xlabel('Profundidad máxima')
plt.ylabel('Precisión en test')
plt.grid(True)
plt.show()

print(f'Máxima precisión alcanzada: {max(accuracies):.3f} con profundidad {max_depths[np.argmax(accuracies)]}')
