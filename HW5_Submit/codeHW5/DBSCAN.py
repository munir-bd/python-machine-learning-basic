import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(len(X))

from sklearn.cluster import AgglomerativeClustering

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

km = KMeans(n_clusters=3, random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 2],
            edgecolor='black',
            c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 2],
            edgecolor='black',
            c='red', marker='s', s=40, label='cluster 2')
ax1.scatter(X[y_km == 2, 0], X[y_km == 2, 2],
            edgecolor='black',
            c='blue', marker='x', s=40, label='cluster 3')

ax1.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 2],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')

ax1.set_title('K-means clustering')



ac = AgglomerativeClustering(n_clusters=3,
                             affinity='euclidean',
                             linkage='complete')
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 2], c='lightblue',
            edgecolor='black',
            marker='o', s=40, label='cluster 1')
ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 2], c='red',
            edgecolor='black',
            marker='s', s=40, label='cluster 2')
ax2.scatter(X[y_ac == 2, 0], X[y_ac == 2, 2], c='blue',
            edgecolor='black',
            marker='x', s=40, label='cluster 3')
# ax2.scatter(km.cluster_centers_[:, 0],
#             km.cluster_centers_[:, 2],
#             s=250, marker='*',
#             c='red', edgecolor='black',
#             label='centroids')
ax2.set_title('Agglomerative clustering')

plt.legend()
plt.tight_layout()
plt.show()



from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.42, min_samples=4.3, metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db == 0, 0], X[y_db == 0, 2],
            c='lightblue', marker='o', s=40,
            edgecolor='black',
            label='cluster 1')
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 2],
            c='red', marker='s', s=40,
            edgecolor='black',
            label='cluster 2')
plt.scatter(X[y_db == 2, 0], X[y_db == 2, 2],
            c='blue', marker='x', s=40,
            edgecolor='black',
            label='cluster 3')

plt.legend()
plt.tight_layout()
plt.show()