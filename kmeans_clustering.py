# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Backup.csv')
X_temp = data.iloc[:, 1:3].values

from sklearn.preprocessing import StandardScaler
sc_X =  StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(X_temp))


# use elbow mwthod to find optimal number of clusters
from sklearn.cluster import KMeans
'''
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters =i, init="k-means++", max_iter=300, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

    
plt.plot(range(1, 11), wcss);
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

'''
# choosen cluster is 4
kmeans = KMeans(n_clusters=4, init="k-means++", max_iter=1000, n_init=10)
y_pred = kmeans.fit_predict(X)


#plot the scatters
plt.scatter(X[y_pred == 0].iloc[:, 0], X[y_pred == 0].iloc[:, 1], s=5, c="red", label="A")
plt.scatter(X[y_pred == 1].iloc[:, 0], X[y_pred == 1].iloc[:, 1], s=5, c="green", label="B")
plt.scatter(X[y_pred == 2].iloc[:, 0], X[y_pred == 2].iloc[:, 1], s=5, c="blue", label="C")
plt.scatter(X[y_pred == 3].iloc[:, 0], X[y_pred == 3].iloc[:, 1], s=5, c="purple", label="D")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c="black", marker="*")

plt.ylim([0,3])
plt.xlabel("Sold Quantity")
plt.ylabel("Unit Price")
plt.show()

