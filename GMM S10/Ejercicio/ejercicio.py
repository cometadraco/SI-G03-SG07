import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

file_path = '/content/SI_L11_GMM_DATASET.csv'
data = pd.read_csv(file_path)

# Convertir la columna de fechas a tipo datetime y eliminarla si es necesario
data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'], errors='coerce')  
data = data.drop(columns=['Unnamed: 0', 'ticker']) 
# Seleccionamos las columnas relevantes para el análisis (ignorar la columna de fecha y ticker)
features = data[['open', 'high', 'low', 'close', 'adjclose', 'volume']]

# Normalizamos los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Aplicamos el modelo GMM
gmm = GaussianMixture(n_components=3)
gmm.fit(scaled_data)

# Predicción de los clusters
clusters = gmm.predict(scaled_data)

data['Cluster'] = clusters

# Graficar los clusters en 2 dimensiones usando los primeros dos componentes principales
plt.figure(figsize=(10, 6))
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap='viridis')
plt.title("Clusters de activos financieros (GMM)")
plt.xlabel("Precio de apertura normalizado")
plt.ylabel("Precio máximo normalizado")
plt.colorbar(label="Cluster")
plt.show()

# Obtener el BIC para determinar el número óptimo de clusters
bic_scores = [GaussianMixture(n_components=i).fit(scaled_data).bic(scaled_data) for i in range(1, 11)]

# Graficamos el BIC para observar el número óptimo de clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), bic_scores, marker='o')
plt.title("Criterio BIC para el número óptimo de clusters")
plt.xlabel("Número de Clusters")
plt.ylabel("BIC")
plt.show()

# Análisis de las características promedio por cluster
data_grouped = data.groupby('Cluster').mean()

# Mostrar las características promedio de cada cluster
print(data_grouped[['open', 'high', 'low', 'close', 'adjclose', 'volume']])