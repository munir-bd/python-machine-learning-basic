import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sns.set()
sns.set_context("talk")

#1) Load (all features, using Pandas), standardize this d-dimension dataset (d is number of features) and Split
# the Iris dataset to training and test sets with ratio 70% and 30%, respectively.
# dictionary of the feature names
feature_dict = {i:label for i,label in zip(
            range(4),
              ('sepal length in cm',
              'sepal width in cm',
              'petal length in cm',
              'petal width in cm', ))}

print(feature_dict)

# reading the CSV file directly from the UCI machine learning repository
df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
    )

df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True) # to drop the empty line at file-end

print(df.tail())

# convert pandas DataFrame to simple numpy arrays
X = df[['sepal length in cm','sepal width in cm','petal length in cm','petal width in cm']].values
# X = df[[0,1,2,3]].values
y = df['class label'].values
# convert class labels from strings to integers
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y)

print("X: ",X)
print("y: ",y)

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

print("X_train_std: ",X_train_std)
print("X_test_std: ",X_test_std)


# 2) Write your own function to calculate the  matrix SW−1 SB  matrix.
# Then compute the eigenvalues and eigenvectors of this matrix.

#Calculate the mean vectors for each class:
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(0, 3):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('MV %s: %s\n' % (label, mean_vecs[label - 1]))
# Compute the within-class scatter matrix:
d = 4  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(0, 3), mean_vecs):
    class_scatter = np.zeros((d, d))  # scatter matrix for each class
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter                          # sum class scatter matrices
print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
print('Class label distribution: %s' % np.bincount(y_train)[0:])

# Compute the between-class scatter matrix:
mean_overall = np.mean(X_train_std, axis=0)
d = 4  # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    mean_overall = mean_overall.reshape(d, 1)  # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))
# Solve the generalized eigenvalue problem for the matrix S−1WSBSW−1SB:
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
print('\nEigenvalues \n%s' % eigen_vals)
print('\neigen_vecs \n%s' % eigen_vecs)
# Sort eigenvectors in decreasing order of the eigenvalues:
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print("eigen_pairs: ",eigen_pairs)
print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print("eigen_val[0]: ",eigen_val[0])

# 3) Plot the cumulative variance ratio (c.f. block [11] of the nbviewer of Chapter 5).
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
print("cum_discr: ",cum_discr)
print("discr: ",discr)
plt.bar(range(0, 4), discr, alpha=0.5, align='center',
        label='individual "discriminability"')
plt.step(range(0, 4), cum_discr, where='mid',label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 4) Choose the k=3 eigenvectors that correspond to the k largest eigenvalues to construct a d×k-dimensional
# transformation matrix W ; the eigenvectors are the columns of this matrix

W = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis], eigen_pairs[2][1][:, np.newaxis]))
print('Matrix W:\n', W)
print("X_train_std[0].dot(W) ",X_train_std[0].dot(W))
X_train_lda = X_train_std.dot(W)
print("X_train_pca ",X_train_lda)

# 5) Project the samples onto the new feature subspace, and plot the projected data using the
# transformation matrix W (c.f. block [14] of the nbviewer of Chapter 5)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
label_dict = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2:'Iris Virginica'}
#LD 1 and LD 2
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1],
                c=c, label=label_dict[l], marker=m)
    print(l)
    print(label_dict[l])
plt.title('LD 1 and LD 2')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='upper center')
plt.tight_layout()
plt.show()

#LD 1 and LD 3
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 2],
                c=c, label=label_dict[l], marker=m)
plt.title('LD 1 and LD 3')
plt.xlabel('LD 1')
plt.ylabel('LD 3')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#LD 2 and LD 3
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 1],
                X_train_lda[y_train == l, 2],
                c=c, label=label_dict[l], marker=m)
plt.title('LD 2 and LD 3')
plt.xlabel('LD 2')
plt.ylabel('LD 3')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


#LD1, LD2 and LD3
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
for l, c, m in zip(np.unique(y_train), colors, markers):
    ax.scatter(X_train_lda[y_train == l, 0], X_train_lda[y_train == l, 1], X_train_lda[y_train == l, 2],c=c,  label=label_dict[l], marker=m, edgecolor='k', s=40)
    print('%s' %label_dict[l])
ax.set_title("First three LD directions")
ax.set_xlabel("LD 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("LD 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("LD 3")
ax.w_zaxis.set_ticklabels([])
ax.legend(loc='upper right')
plt.show()