# Se sube el archivo CSV desde tu computadora
from google.colab import files
uploaded = files.upload()

#Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Cargar el archivo CSV
df = pd.read_csv("SI_L05_REGRESIÓN_LOGÍSTICA_DATASET.csv")

#Limpiar nombres de columnas (elimina espacios y pone minúsculas)
df.columns = df.columns.str.strip().str.lower()
print("Columnas detectadas:", df.columns)

# Verificar y limpiar datos
df = df.dropna()

#Separar variables predictoras (X) y variable objetivo (y)
X = df[['clump thickness', 'uniformity of cell size']]
y = df['class']  # Esta es la variable objetivo

#Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Crear y entrenar el modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Evaluacion del modelo
y_pred = modelo.predict(X_test)
print(f"\nPrecisión del modelo: {accuracy_score(y_test, y_pred):.2f}")
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# Mostrar matriz de confusión
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

#Visualizar frontera de decisión
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.xlabel('Edad (Normalizada)')
    plt.ylabel('Salario (Normalizado)')
    plt.title('Frontera de Decisión')
    plt.show()

plot_decision_boundary(X_train, y_train, modelo)
