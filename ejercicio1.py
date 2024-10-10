import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo de postulantes
postulantes_df = pd.read_excel('BI_Postulantes09-1.xlsx')

# Seleccionar las columnas numéricas para K-means
numeric_cols = postulantes_df[['Apertura Nuevos Conoc.', 'Nivel Organización', 
                               'Participación Grupo Social', 'Grado Empatía', 
                               'Grado Nerviosismo', 'Dependencia Internet']]

# Aplicar K-means con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
postulantes_df['Cluster'] = kmeans.fit_predict(numeric_cols)

# Visualización: histogramas cruzando la columna 'Nom_Especialidad' con los clusters
plt.figure(figsize=(10, 6))
sns.histplot(data=postulantes_df, x='Nom_Especialidad', hue='Cluster', multiple='stack')
plt.title('Histograma de Especialidad cruzado con Clusters')
plt.xlabel('Especialidad')
plt.ylabel('Frecuencia')
plt.xticks(rotation=90)  # Rotar etiquetas si hay muchas especialidades
plt.show()

