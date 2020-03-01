import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


sns.set()
sns.set_context("talk")

#1) Load (all features, using Pandas), standardize this d-dimension dataset (d is number of features) and Split
# the Iris dataset to training and test sets with ratio 70% and 30%, respectively.

iris_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print("iris_df.head(): ",iris_df.head())
print("iris_df.tail(): ",iris_df.tail())

X = iris_df.iloc[:,0:4].values
y = iris_df.iloc[:,4].values

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 2) Write your own function to calculate the covariance matrix. Then compute the eigenvalues and
# eigenvectors of this matrix.
mean_vec = np.mean(X_train_std, axis=0)
cov_mat = (X_train_std - mean_vec).T.dot((X_train_std - mean_vec)) / (X_train_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

# 3) Plot the cumulative variance ratio (c.f. block [11] of the nbviewer of Chapter 5).
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(0, 4), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(0, 4), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='center right')
plt.tight_layout()
plt.show()


# 4) Choose the k=3 eigenvectors that correspond to the k largest eigenvalues to construct a d×k-dimensional
# transformation matrix W ; the eigenvectors are the columns of this matrix
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
print(eigen_pairs)
W = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis], eigen_pairs[2][1][:, np.newaxis]))
print('Matrix W:\n', W)
print("X_train_std[0].dot(W) ",X_train_std[0].dot(W))

# 5) Project the samples onto the new feature subspace, and plot the projected data using the
# transformation matrix W (c.f. block [14] of the nbviewer of Chapter 5)
X_train_pca = X_train_std.dot(W)
print("X_train_pca ",X_train_pca)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
#PCA 1 and PCA2
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0],
                X_train_pca[y_train == l, 1],
                c=c, label=l, marker=m)
plt.title('PC 1 and PC 2')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

#PCA 1 and PCA 3
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0],
                X_train_pca[y_train == l, 2],
                c=c, label=l, marker=m)
plt.title('PC 1 and PC 3')
plt.xlabel('PC 1')
plt.ylabel('PC 3')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

#PCA 2 and PCA 3
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 1],
                X_train_pca[y_train == l, 2],
                c=c, label=l, marker=m)
plt.title('PC 2 and PC 3')
plt.xlabel('PC 2')
plt.ylabel('PC 3')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
for l, c, m in zip(np.unique(y_train), colors, markers):
    ax.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], X_train_pca[y_train == l, 2],c=c,   label=l,marker=m, edgecolor='k', s=40)

ax.set_title("First three PC directions")
ax.set_xlabel("PC 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("PC 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("PC 3")
ax.w_zaxis.set_ticklabels([])
plt.legend(loc='upper right')

plt.show()




