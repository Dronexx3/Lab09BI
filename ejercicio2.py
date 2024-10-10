import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Cargar el archivo de clientes
clientes_df = pd.read_excel('BI_Clientes09-1.xlsx')

# Convertir columnas categóricas a variables numéricas usando get_dummies
clientes_df_encoded = pd.get_dummies(clientes_df[['YearlyIncome', 'TotalChildren', 'NumberChildrenAtHome', 'EnglishEducation', 'Age']])

# Definir la variable objetivo
y = clientes_df['BikeBuyer']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(clientes_df_encoded, y, test_size=0.3, random_state=42)

# Crear el modelo de árbol de decisiones
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Visualizar el árbol de decisiones
plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['No', 'Yes'])
plt.show()

