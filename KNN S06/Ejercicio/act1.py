# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Generar 100 muestras de clientes ficticios
num_samples = 100
minutos_mensuales = np.random.randint(100, 600, num_samples)  # Minutos hablados entre 100 y 600
mensajes_texto = np.random.randint(10, 200, num_samples)      # SMS enviados entre 10 y 200
uso_datos = np.random.uniform(0.5, 10, num_samples)           # Uso de datos en GB entre 0.5 y 10

# Calcular facturación mensual estimada
facturacion_mensual = minutos_mensuales * 0.1 + mensajes_texto * 0.05 + uso_datos * 5

# Generar variable objetivo (Churn: 1=se va, 0=se queda) según facturación
churn = (facturacion_mensual > 50).astype(int)

# Crear DataFrame
data = pd.DataFrame({
    'Minutos Mensuales': minutos_mensuales,
    'Mensajes de Texto': mensajes_texto,
    'Uso de Datos (GB)': uso_datos,
    'Facturación Mensual': facturacion_mensual,
    'Churn': churn
})

# Mostrar primeras filas, info y descripción estadística
print(data.head())
print(data.info())
print(data.describe())

# Separar características (X) y variable objetivo (y)
X = data.drop(columns=['Churn'])
y = data['Churn']

# Dividir en conjunto de entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Evaluar diferentes valores de K para KNN
k_values = range(1, 21)
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_k))

# Graficar precisión en función de K
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel("Número de Vecinos (K)")
plt.ylabel("Precisión del Modelo")
plt.title("Efecto del Número de Vecinos (K) en la Precisión de KNN")
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Seleccionar el mejor valor de K
best_k = k_values[np.argmax(accuracies)]
print(f"El mejor valor de K encontrado: {best_k}")

# Entrenar y evaluar modelo final con el mejor K
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Precisión del modelo con K={best_k}: {accuracy:.2f}')
print("Matriz de Confusión:\n", conf_matrix)
print("Reporte de Clasificación:\n", class_report)

# Gráfico de dispersión: Uso de Datos vs Facturación Mensual, coloreado por Churn
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=data['Uso de Datos (GB)'],
    y=data['Facturación Mensual'],
    hue=data['Churn'],
    palette='coolwarm',
    alpha=0.7
)
plt.xlabel("Uso de Datos (GB)")
plt.ylabel("Facturación Mensual ($)")
plt.title("Gráfico de Dispersión: Facturación Mensual vs Uso de Datos")
plt.legend(title="Churn (0=Se queda, 1=Se va)")
plt.show()
