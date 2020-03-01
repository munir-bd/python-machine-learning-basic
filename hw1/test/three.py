import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import seaborn as sns
sns.set()

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




C1=[10.0, 100.0,1000.0,5000.0]






import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
X_xor = np.random.randn(700, 2)
X_test = np.random.randn(300, 2)

y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)

y_test = np.logical_xor(X_test[:, 0] > 0,
                        X_test[:, 1] > 0)


y_xor = np.where(y_xor, 1, -1)
y_test = np.where(y_test, 1, -1)


plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.title("Training Set")
plt.tight_layout()
# plt.savefig('./figures/xor.png', dpi=300)
plt.show()



plt.scatter(X_test[y_test == 1, 0],
            X_test[y_test == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_test[y_test == -1, 0],
            X_test[y_test == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.title("Test Set")
plt.tight_layout()
# plt.savefig('./figures/xor.png', dpi=300)
plt.show()

from sklearn.svm import SVC
for cc in C1:

    svm = SVC(kernel='rbf', random_state=0, gamma=0.20, C=cc)
    svm.fit(X_xor, y_xor)

    # draw decision boundary
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    # plt.title("C=,cc 10, gamma=0.2")
    plt.title(cc)
    plt.show()










svm = SVC(kernel='rbf', random_state=0, gamma=0.20, C=cc)
svm.fit(X_xor, y_xor)

# draw decision boundary
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.title("C=10, gamma=0.2")
plt.show()


from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=0.20, C=100.0)
svm.fit(X_xor, y_xor)

# draw decision boundary
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.title("C=100, gamma=0.2")
plt.show()


from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=0.20, C=1000.0)
svm.fit(X_xor, y_xor)

# draw decision boundary
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.title("C=1000, gamma=0.2")
plt.show()


from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=0.20, C=5000.0)
svm.fit(X_xor, y_xor)

# draw decision boundary
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.title("C=5000, gamma=0.2")
plt.show()



from sklearn.metrics import accuracy_score

for cc in C1:
    print(cc)

    accuracy = []
    weights, params = [], []
    for c in np.arange(0.1, 100.):
        svm = SVC(kernel='rbf', random_state=0, gamma=c, C=cc)
        svm.fit(X_xor, y_xor)
        # weights.append(svm.coef_[1])
        params.append(c)
        y_pred = svm.predict(X_test)
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
        accuracy.append(accuracy_score(y_test, y_pred))

    weights = np.array(weights)
    # plt.plot(params, weights[:, 0],label='sepal length')
    # plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
    plt.plot(params, accuracy, color='b', linestyle='--',label=cc)
    plt.ylabel('Accuracy')
    plt.xlabel('gamma')
    plt.legend(loc='bottom left')
    # plt.title("comparison")
    plt.xscale('log')
        # plt.savefig('./figures/regression_path.png', dpi=300)
    plt.show()


for cc in C1:
    print(cc)

    accuracy = []
    weights, params = [], []
    for c in np.arange(0.1, 100.):
        svm = SVC(kernel='rbf', random_state=0, gamma=c, C=cc)
        svm.fit(X_xor, y_xor)
        # weights.append(svm.coef_[1])
        params.append(c)
        y_pred = svm.predict(X_test)
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
        accuracy.append(accuracy_score(y_test, y_pred))

    weights = np.array(weights)
    # plt.plot(params, weights[:, 0],label='sepal length')
    # plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
    plt.plot(params, accuracy, color='b', linestyle='--',label=cc)
plt.ylabel('Accuracy')
plt.xlabel('gamma')
plt.legend(loc='bottom left')
plt.title("comparison")
plt.xscale('log')
    # plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()

