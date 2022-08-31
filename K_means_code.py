# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:51:11 2021

@author: monte
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()


#QUITAMOS LA COLUMNA ADDRESS PORQUE ES CATEORICA Y NO TIENE SENTIDO CALCULAR LA DISTACIA EUCLIDIANA
df = cust_df.drop('Address', axis=1)
df.head()

#NORMALIZAMOS EL DATASET
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

#Aplicacmos el metodo de k-medias
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


df["Clus_km"] = labels
df.head(5)

#podemos obtener la ubicaion de los centroides
df.groupby('Clus_km').mean()

#Ahora, miremos la distribuición de los clientes basados en su edad e ingreso:
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()



#REALIZAMOS UNA GRAFICA 3D DE LA 

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))