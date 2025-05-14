import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('SI_L06_KNN_DATASET.csv')

# Imputar ceros con mediana en columnas críticas
cols_zero_not_valid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_zero_not_valid:
    median_val = data[data[col] != 0][col].median()
    data[col] = data[col].replace(0, median_val)

# Gráfico de dispersión (Glucose vs BMI con Outcome)
plt.figure(figsize=(8,6))
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=data, palette='coolwarm')
plt.title('Gráfico de Dispersión: Glucosa vs BMI por Outcome')
plt.xlabel('Nivel de Glucosa')
plt.ylabel('Índice de Masa Corporal (BMI)')
plt.legend(title='Outcome (0=No diabetes, 1=Diabetes)')
plt.show()

# Preparar datos para modelo
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Escalar variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Encontrar K óptimo con tasa de error
k_range = list(range(1, 41))
mean_errors = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error = np.mean(pred != y_test)
    mean_errors.append(error)

# Gráfico de tasa de error vs K
plt.figure(figsize=(8,6))
plt.plot(k_range, mean_errors, marker='o', linestyle='--', color='b')
plt.title('Error Rate Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Gráfico de accuracy vs K
scores = [1 - err for err in mean_errors]
plt.figure(figsize=(8,6))
plt.plot(k_range, scores, marker='o', linestyle='-', color='green')
plt.title('Exactitud vs K')
plt.xlabel('Número de vecinos K')
plt.ylabel('Exactitud (Accuracy)')
plt.xticks(k_range)
plt.grid(True)
plt.show()

best_k = k_range[np.argmin(mean_errors)]
print(f"El K óptimo según tasa de error es: {best_k} con error: {min(mean_errors):.4f} y accuracy: {scores[np.argmin(mean_errors)]:.4f}")

# Entrenar modelo final con K óptimo
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred = knn_final.predict(X_test)

# Reporte clasificación
print("\nReporte de clasificación para KNN:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='magma')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# Comparar con otros modelos
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=best_k)
}

results = {}
print("\nAccuracy de otros modelos:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

plt.figure(figsize=(8,6))
plt.bar(results.keys(), results.values())
plt.ylabel('Accuracy')
plt.title('Comparación de modelos')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()
