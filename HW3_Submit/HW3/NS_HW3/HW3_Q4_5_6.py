import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np
if __name__ == '__main__':
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                          'machine-learning-databases/wine/wine.data',
                          header=None)

    # if the Breast Cancer dataset is temporarily unavailable from the
    # UCI machine learning repository, un-comment the following line
    # of code to load the dataset from a local path:

    # df_wine = pd.read_csv('wdbc.data', header=None)

    df_wine.head()


    from sklearn.preprocessing import LabelEncoder

    X = df_wine.loc[:, 1:].values
    y = df_wine.loc[:, 0].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         test_size=0.30,
                         stratify=y,
                         random_state=1)

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline


    # 4)  Using GridSearchCV to find the best hyperparameter (similar to [17] of this notebook).
    # Compare the accuracy score using these GridSearchCV parameters with previous methods.

    # 5) Report the confusion matrix of the above prediction model using GridSearchCV.

    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    pipe_svc = make_pipeline(StandardScaler(),
                             SVC(random_state=1))

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    param_grid = [{'svc__C': param_range,
                   'svc__kernel': ['linear']},
                  {'svc__C': param_range,
                   'svc__gamma': param_range,
                   'svc__kernel': ['rbf']}]

if __name__ == '__main__':
    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)

    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    print('Test accuracy: %.3f' % clf.score(X_test, y_test))

    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=2)

    scores = cross_val_score(gs, X_train, y_train,
                             scoring='accuracy', cv=10)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                          np.std(scores)))
# 5 Report the confusion matrix of the above prediction model using GridSearchCV.
    from sklearn.metrics import confusion_matrix

    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.tight_layout()
    plt.show()
#6 Report the precision and recall scores as in [29] and the best scores and best parameter of GridSearchCV as in [30] of this notebook.

    from sklearn.metrics import precision_score, recall_score, f1_score
    # will return the total ratio of tp/(tp + fp)
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='micro'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average='micro'))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average='micro'))

    from sklearn.metrics import make_scorer

    scorer = make_scorer(f1_score, pos_label=0)

    c_gamma_range = [0.01, 0.1, 1.0, 10.0]

    param_grid = [{'svc__C': c_gamma_range,
                   'svc__kernel': ['linear']},
                  {'svc__C': c_gamma_range,
                   'svc__gamma': c_gamma_range,
                   'svc__kernel': ['rbf']}]

    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring=scorer,
                      cv=10,
                      n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)











