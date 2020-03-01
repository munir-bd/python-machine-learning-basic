import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)

# df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
#                    'Alcalinity of ash', 'Magnesium', 'Total phenols',
#                    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
#                    'Color intensity', 'Hue',
#                    'OD280/OD315 of diluted wines', 'Proline']

# print(df_wine.head())
#
# print(df_wine.tail())

# X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
#
# sc = StandardScaler()
# X_train_std = sc.fit_transform(X_train)
# X_test_std = sc.transform(X_test)

print(df_wine.shape)
from sklearn.preprocessing import LabelEncoder
X = df_wine.loc[:, 1:].values
y = df_wine.loc[:, 0].values
le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)

# 1) Split dataset into ratio 7:3 for training and test sets, respectively.
# Then use pipeline with StandardScaler(), PCA (n=3), and SVM with RBF kernel to fit the
# training set and predict the test set. Report the accuracy score.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# pipe_lr = make_pipeline(StandardScaler(),PCA(n_components=2), LogisticRegression(random_state=1))
# pipe_lr.fit(X_train, y_train)
# y_pred = pipe_lr.predict(X_test)
# print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))


from sklearn.svm import SVC
pipe_lr = make_pipeline(StandardScaler(),PCA(n_components=3), SVC(random_state=1))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

# 2) Use StratifiedKFold cross-validation to report the accuracy score (mean with std).
# What are differences between standard k-fold and StratifiedKFold?
# What are differences between StratifiedKFold and cross_val_score (in [12] and [13] of this notebook)?

# KFold will divide your data set into prespecified number of folds, and every sample must be in one and only one fold.
# A fold is a subset of your dataset.
#
# ShuffleSplit will randomly sample your entire dataset during each iteration to generate a training set and a test set.
# The test_size and train_size parameters control how large the test and training test set should be for each iteration.
# Since you are sampling from the entire dataset during each iteration, values selected during one iteration, could be selected again during another iteration.
#
# Summary: ShuffleSplit works iteratively, KFold just divides the dataset into k folds.
#
# Difference when doing validation
#
# In KFold, during each round you will use one fold as the test set and all the remaining folds as your training set.
# However, in ShuffleSplit, during each round n you should only use the training and test set from iteration n.
# As your data set grows, cross validation time increases, making shufflesplits a more attractive alternate.
# If you can train your algorithm, with a certain percentage of your data as opposed to using all k-1 folds, ShuffleSplit is an attractive option.

from sklearn.model_selection import KFold
from sklearn import svm
svc = svm.SVC(C=1, kernel='linear')
kf = KFold(n_splits=10)
kf.get_n_splits(X)
# print(kf)
KF_scores = list()
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    KF_scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print('\nKF_scores: %s' % KF_scores)
print('KF accuracy: %.3f +/- %.3f' % (np.mean(KF_scores), np.std(KF_scores)))


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10,random_state=1).split(X_train, y_train)
StratifiedKFold_scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    StratifiedKFold_scores.append(score)
    print('StratifiedFold: %2d, Class dist.: %s, Acc: %.3f' % (k + 1,np.bincount(y_train[train]), score))

print('StratifiedKFold CV accuracy: %.3f +/- %.3f' % (np.mean(StratifiedKFold_scores), np.std(StratifiedKFold_scores)))


from sklearn.model_selection import cross_val_score

CV_scores = cross_val_score(estimator=pipe_lr, X=X_train,y=y_train, cv=10, n_jobs=1)
print('\nCV accuracy scores: %s' % CV_scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(CV_scores), np.std(CV_scores)))

sns.set()
sns.set_context("talk")
plt.plot(KF_scores,color='r',lw = 2.0, linestyle='--',label='standard k-fold')
plt.plot(StratifiedKFold_scores,color='g', lw = 2.0, linestyle=':',label='StratifiedKFold')
plt.ylabel('Scores', fontsize = 16)
plt.xlabel('# of Fold', fontsize = 16)
plt.title("standard k-fold Vs. StratifiedKFold")
plt.legend(loc='best',fontsize = 14)
plt.show()


plt.plot(CV_scores,color='r',lw = 2.0, linestyle='--',label='CV_scores')
plt.plot(StratifiedKFold_scores,color='g', lw = 2.0, linestyle=':',label='StratifiedKFold')
plt.ylabel('Scores', fontsize = 16)
plt.xlabel('# of Fold', fontsize = 16)
plt.title("CV_scores Vs. StratifiedKFold")
plt.legend(loc='best',fontsize = 14)
plt.show()




