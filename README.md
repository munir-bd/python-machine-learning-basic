# python-machine-learning-basic

In order to develop all of the exercises of this repo, the instructor follows Python Machine Learning 2nd Edition by Sebastian Raschka, Packt Publishing Ltd. 2017 as a reference book.

Homework1: Logistic Regression and SVM
 

I/ Logistic Regression v.s. SVM with Iris dataset:
 1) Load (all features) and Split the Iris dataset to training and test sets with ratio 80% and 20%, respectively (c.f. block [4] and [5] of the nbviewer of Chapter 3). 


 2) Fit your data using Logistic Regression in sklearns. Then plot the decision regions with C=10,100,1000,5000 (generate one figure for each value of C). The x-axis and y-axis of each figure correspond to feature "sepal length" and "petal length", correspondingly (c.f. block [16] of the nbviewer of Chapter 3). 

 3) Plot the accuracy_score (in sklearns) of your predictor with respect to C=10−x,x=−4,−3,...,3,4. 

 4) Repeat steps 1~3 above, using SVM instead of logistic regression, with kernel "RBF", for two cases of  γ=0.1 and 10. 

 5) Compare two types of predictor (give insight comments)

II/ SVM for XOR data:

 1) Generate 1000 random samples of XOR data (c.f. block [23] of nbviewer of Chapter 3). Split 70% and 30% for training and test sets. 

 2) Use SVM to fit your data with kernel "RBF" and γ=0.2. Then plot the decision regions with C=10,100,1000,5000 (generate one figure for each value of C). 

 3) Plot the accuracy score with respect to γ=0.1,…,100
 
 HW2: PCA and LDA
I/ PCA Iris dataset:
 1) Load (all features, using Pandas), standardize this d-dimension dataset (d is number of features) and Split the Iris dataset to training and test sets with ratio 70% and 30%, respectively. 

 2) Write your own function to calculate the covariance matrix. Then compute the eigenvalues and eigenvectors of this matrix. 

 3) Plot the cumulative variance ratio (c.f. block [11] of the nbviewer of Chapter 5). 

  4) Choose the k=3 eigenvectors that correspond to the k largest eigenvalues to construct a d×k-dimensional transformation matrix W ; the eigenvectors are the columns of this matrix 

  5) Project the samples onto the new feature subspace, and plot the projected data using the transformation matrix W (c.f. block [14] of the nbviewer of Chapter 5)

 

 

II/ LDA for Iris:

  Repeat the steps above of PCA, but using the matrix S−1WSB instead of covariance matrix. 

