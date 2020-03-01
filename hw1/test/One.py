import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target
print('Class labels:', np.unique(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

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

# for cc in C1:
#     lr1 = LogisticRegression(C=cc, random_state=0)
#     lr1.fit(X_train_std, y_train)
#
#     X_combined_std = np.vstack((X_train_std, X_test_std))
#     y_combined = np.hstack((y_train, y_test))
#
#     plot_decision_regions(X_combined_std, y_combined, classifier=lr1, test_idx=range(105, 150))
#     # plt.xlabel('petal length [standardized]')
#     # plt.ylabel('petal width [standardized]')
#
#     plt.xlabel('sepal length [standardized]')
#     plt.ylabel('petal length [standardized]')
#     plt.title(cc)
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#     # plt.savefig('./figures/logistic_regression.png', dpi=300)
#     plt.show()


#10
#
lr1 = LogisticRegression(C=10.0, random_state=0)
lr1.fit(X_train_std, y_train)


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=lr1, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')

plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title("C=10")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/logistic_regression.png', dpi=300)
plt.show()


lr2 = LogisticRegression(C=100.0, random_state=0)
lr2.fit(X_train_std, y_train)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=lr2, test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.title("C=100")
plt.tight_layout()
# plt.savefig('./figures/logistic_regression.png', dpi=300)
plt.show()


lr3 = LogisticRegression(C=1000.0, random_state=0)
lr3.fit(X_train_std, y_train)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=lr3, test_idx=range(105, 150))

plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title("C=1000")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/logistic_regression.png', dpi=300)
plt.show()



lr4 = LogisticRegression(C=5000.0, random_state=0)
lr4.fit(X_train_std, y_train)


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=lr4, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')

plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title("C=5000")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/logistic_regression.png', dpi=300)
plt.show()

import seaborn as sns
sns.set()

from sklearn.metrics import accuracy_score
accuracy3 = []
weights, params = [], []
for c in np.arange(-4., 4.):
    lr4 = LogisticRegression(C=10.**c, random_state=0)
    lr4.fit(X_train_std, y_train)
    weights.append(lr4.coef_[1])
    params.append(10**c)
    y_pred = lr4.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy3.append(accuracy_score(y_test, y_pred))

plt.plot(params, accuracy3,color='r', linestyle='--',label='Predictor Accuracy')


plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='bottom left')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()

