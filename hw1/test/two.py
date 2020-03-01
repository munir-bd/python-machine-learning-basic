import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

df = pd.read_csv('data1.csv', header=None)

print(df.tail())
#
# X = df.data[:, [2, 3]]
# y = df.target
# print('Class labels:', np.unique(y))

# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class:
# -- Iris Setosa
# -- Iris Versicolour
# -- Iris Virginica

import seaborn as sns
sns.set()


iris = datasets.load_iris()
# print(iris.tail())
# X = iris.data[:, [2, 3]]
X = iris.data[:, [0, 2]]
y = iris.target
print('Class labels:', np.unique(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)




from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


from sklearn.svm import SVC


C1=[10.0, 100.0,1000.0,5000.0]

# for cc in C1:
#     svm1 = SVC(kernel='rbf', random_state=0, gamma=0.1, C=cc)
#     svm1.fit(X_train_std, y_train)
#
#     X_combined_std = np.vstack((X_train_std, X_test_std))
#     y_combined = np.hstack((y_train, y_test))
#
#     plot_decision_regions(X_combined_std, y_combined, classifier=svm1, test_idx=range(105, 150))
#     plt.xlabel('sepal length [standardized]')
#     plt.ylabel('petal length [standardized]')
#     plt.title(cc)
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#     # plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
#     plt.show()

#
#
# #SVM 10
svm1 = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
svm1.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm1, test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title("C=10, gamma=0.1")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
plt.show()

# #SVM 100
svm2 = SVC(kernel='rbf', random_state=0, gamma=0.1, C=100.0)
svm2.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm2, test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title("C=100, gamma=0.1")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
plt.show()




# #SVM 1000
svm3 = SVC(kernel='rbf', random_state=0, gamma=0.1, C=1000.0)
svm3.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm3, test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title("C=1000, gamma=0.1")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
plt.show()



# #SVM 5000
svm4 = SVC(kernel='rbf', random_state=0, gamma=0.1, C=5000.0)
svm4.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm4, test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title("C=5000, gamma=0.1")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
plt.show()




from sklearn.metrics import accuracy_score
accuracy1 = []
weights, params = [], []
for c in np.arange(-4., 4.):
    svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.**c)
    svm.fit(X_train_std, y_train)
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy1.append(accuracy_score(y_test, y_pred))

plt.plot(params, accuracy1,color='r', linestyle=':',label='Predictor Accuracy Gamma=0.1')


# accuracy2 = []
# # 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
# svm2.predict(X_test_std[0, :].reshape(1, -1))
# weights, params = [], []
# for c in np.arange(-4., 4.):
#     # lr = LogisticRegression(C=10.**c, random_state=0)
#     svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
#     svm.fit(X_train_std, y_train)
#     # weights.append(svm.coef_[1])
#     params.append(10**c)
#     y_pred = svm.predict(X_test_std)
#     print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     accuracy2.append(accuracy_score(y_test, y_pred))
#
# plt.plot(params, accuracy2,color='b', linestyle=':',label='predictor 2')
#
#
# accuracy3 = []
# # 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
# svm3.predict(X_test_std[0, :].reshape(1, -1))
# weights, params = [], []
# for c in np.arange(-4., 4.):
#     # lr = LogisticRegression(C=10.**c, random_state=0)
#     svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
#     svm.fit(X_train_std, y_train)
#     # weights.append(svm.coef_[1])
#     params.append(10**c)
#     y_pred = svm.predict(X_test_std)
#     print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     accuracy3.append(accuracy_score(y_test, y_pred))
#
# plt.plot(params, accuracy3,color='g', linestyle=':',label='predictor 3')
#
#
# accuracy4 = []
# # 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
# # svm4.predict(X_test_std[0, :].reshape(1, -1))
# weights, params = [], []
# for c in np.arange(-4., 4.):
#     # lr = LogisticRegression(C=10.**c, random_state=0)
#     svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
#     svm.fit(X_train_std, y_train)
#     # weights.append(svm.coef_[1])
#     params.append(10**c)
#     y_pred = svm.predict(X_test_std)
#     print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     accuracy4.append(accuracy_score(y_test, y_pred))
#
# plt.plot(params, accuracy4,color='y', linestyle=':',label='predictor 4')

plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='bottom left')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()



C1=[10.0, 100.0,1000.0,5000.0]
#
# for cc in C1:
#     svm5 = SVC(kernel='rbf', random_state=0, gamma=10.0, C=cc)
#     svm5.fit(X_train_std, y_train)
#
#     X_combined_std = np.vstack((X_train_std, X_test_std))
#     y_combined = np.hstack((y_train, y_test))
#
#     plot_decision_regions(X_combined_std, y_combined, classifier=svm5, test_idx=range(105, 150))
#     plt.xlabel('sepal length [standardized]')
#     plt.ylabel('petal length [standardized]')
#     plt.title(cc)
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#     # plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
#     plt.show()


# #SVM 10
svm5 = SVC(kernel='rbf', random_state=0, gamma=10.0, C=10.0)
svm5.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm5, test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title("C=10, gamma=10.0")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
plt.show()




# #SVM 100
svm6 = SVC(kernel='rbf', random_state=0, gamma=10, C=100.0)
svm6.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm6, test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title("C=100, gamma=10.0")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
plt.show()




# #SVM 1000
svm7 = SVC(kernel='rbf', random_state=0, gamma=10, C=1000.0)
svm7.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm7, test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title("C=1000, gamma=10.0")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
plt.show()



# #SVM 5000
svm8 = SVC(kernel='rbf', random_state=0, gamma=10, C=5000.0)
svm8.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm8, test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title("C=5000, gamma=10.0")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
plt.show()


accuracy4 = []
weights, params = [], []
for c in np.arange(-4., 4.):
    svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
    svm.fit(X_train_std, y_train)
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy4.append(accuracy_score(y_test, y_pred))

plt.plot(params, accuracy4,color='b', linestyle=':',label='Predictor Accuracy Gamma=10')

plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='bottom left')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()