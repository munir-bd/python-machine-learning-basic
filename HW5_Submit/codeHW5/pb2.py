import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

np.random.seed(5)
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(len(X))
variables = ['W','X', 'Y', 'Z']
df = pd.DataFrame(X, columns=variables)
print(df)
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), )
print(row_dist)

from scipy.cluster.hierarchy import linkage

row_clusters = linkage(row_dist, method='complete', metric='euclidean')
pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1)
                    for i in range(row_clusters.shape[0])])

print(row_clusters)

row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1)
                    for i in range(row_clusters.shape[0])])

print(row_clusters)

from scipy.cluster.hierarchy import dendrogram
row_dendr = dendrogram(row_clusters, )
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()



# plot row dendrogram
fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])

row_dendr = dendrogram(row_clusters, orientation='left')
# reorder data with respect to clustering
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
axd.set_xticks([])
axd.set_yticks([])
# remove axes spines from dendrogram
for i in axd.spines.values():
    i.set_visible(False)
# plot heatmap
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])  # x-pos, y-pos, width, height
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')

fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
axm.set_aspect('auto')
plt.show()




#
# from scipy.spatial.distance import pdist, squareform
#
# row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
#                         columns=labels,
#                         index=labels)
# print(row_dist)
#
#
# from scipy.cluster.hierarchy import linkage
#
# row_clusters = linkage(row_dist, method='complete', metric='euclidean')
# pd.DataFrame(row_clusters,
#              columns=['row label 1', 'row label 2',
#                       'distance', 'no. of items in clust.'],
#              index=['cluster %d' % (i + 1)
#                     for i in range(row_clusters.shape[0])])
#
#
# print("incorrect approach:\n",row_clusters)
#
#
# row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
# pd.DataFrame(row_clusters,
#              columns=['row label 1', 'row label 2',
#                       'distance', 'no. of items in clust.'],
#              index=['cluster %d' % (i + 1)
#                     for i in range(row_clusters.shape[0])])
#
# print("correct approach:\n",row_clusters)
#
# from scipy.cluster.hierarchy import dendrogram
#
# row_dendr = dendrogram(row_clusters,
#                        labels=labels,
#                        # make dendrogram black (part 2/2)
#                        # color_threshold=np.inf
#                        )
# plt.tight_layout()
# plt.ylabel('Euclidean distance')
# #plt.savefig('images/11_11.png', dpi=300,
# #            bbox_inches='tight')
# plt.show()
#
#
#
# # plot row dendrogram
# fig = plt.figure(figsize=(8, 8), facecolor='white')
# axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
#
# # note: for matplotlib < v1.5.1, please use orientation='right'
# row_dendr = dendrogram(row_clusters, orientation='left')
#
# # reorder data with respect to clustering
# df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
#
# axd.set_xticks([])
# axd.set_yticks([])
#
# # remove axes spines from dendrogram
# for i in axd.spines.values():
#     i.set_visible(False)
#
# # plot heatmap
# axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])  # x-pos, y-pos, width, height
# cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
# fig.colorbar(cax)
# axm.set_xticklabels([''] + list(df_rowclust.columns))
# axm.set_yticklabels([''] + list(df_rowclust.index))
#
# #plt.savefig('images/11_12.png', dpi=300)
# plt.show()