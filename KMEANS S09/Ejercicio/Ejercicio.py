# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargar el archivo CSV desde Google Colab
from google.colab import files
uploaded = files.upload()

# Cargar el archivo CSV
data = pd.read_csv(next(iter(uploaded)))

# Seleccionar las columnas relevantes para la segmentación
features = data[['wins', 'kills', 'kdRatio', 'level', 'scorePerMinute', 'gamesPlayed']]

# Normalizar las características
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Método del codo para determinar el número óptimo de clusters
inertia = [] 
range_values = range(1, 11) 

for k in range_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

# Graficar el codo
plt.figure(figsize=(8, 5))
plt.plot(range_values, inertia, marker='o')
plt.title("Método del Codo para Determinar el Número Óptimo de Clusters")
plt.xlabel("Número de Clusters")
plt.ylabel("Inercia")
plt.xticks(range_values)
plt.grid(True)
plt.show()

# Aplicar K-Means con 3 clusters (según el análisis del codo)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Añadir los clusters al DataFrame original
data['cluster'] = clusters

# Calcular las características medias para cada grupo
cluster_summary = data.groupby('cluster')[['wins', 'kills', 'kdRatio', 'level', 'scorePerMinute', 'gamesPlayed']].mean()

# Mostrar el resumen de los clusters
print(cluster_summary)

# 1. Gráfico de dispersión con los clusters
plt.figure(figsize=(10, 6))
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title("Distribución de Jugadores por Clúster (Victorias vs Asesinatos)")
plt.xlabel("Victorias normalizadas")
plt.ylabel("Asesinatos normalizados")
plt.colorbar(label="Cluster")
plt.show()

# 2. Gráfico de barras con las características promedio de cada cluster
cluster_summary.plot(kind='bar', figsize=(12, 8))
plt.title("Promedio de Características por Cluster")
plt.ylabel("Valor promedio")
plt.xlabel("Cluster")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# 3. Graficar los centroides de los clusters
centroids = kmeans.cluster_centers_

plt.figure(figsize=(10, 6))
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroides')
plt.title("Centroides de los Clústeres")
plt.xlabel("Victorias normalizadas")
plt.ylabel("Asesinatos normalizados")
plt.colorbar(label="Cluster")
plt.legend()
plt.show()
