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
  
HW3: Model Evaluation and Hyperparameter Tuning
In this HW, we will use wine dataset, which can be loaded as input [4] of this notebook

 
1) Split dataset into ratio 7:3 for training and test sets, respectively. Then use pipeline with StandardScaler(), PCA (n=3), and SVM with RBF kernel to fit the training set and predict the test set. Report the accuracy score. 
 

2) Use StratifiedKFold cross-validation to report the accuracy score (mean with std). What are differences between standard k-fold and StratifiedKFold? What are differences between StratifiedKFold and cross_val_score (in [12] and [13] of this notebook)?

 
3) What are the differences between "learning curve" and "validation curve" tools in sklearns.model_selection? Report the figures of learning curve and validation curve similar to input [15] and [16] of this notebook, respectively. Based on the figure, indicate which is the best value C you should choose. 
 

4)  Using GridSearchCV to find the best hyperparameter (similar to [17] of this notebook). Compare the accuracy score using these GridSearchCV parameters with previous methods. 

 
5) Report the confusion matrix of the above prediction model using GridSearchCV. 
 

6) Report the precision and recall scores as in [29] and the best scores and best parameter of GridSearchCV as in [30] of this notebook. 

 
7) Plotting the ROCs for every pair combination of classes as in [31] of this notebook, or use the 3 dimension ROCs if it is available.

HW4: Sentiment Analysis
For sentiment analysis, apply similar the technique starting from input [24] of this notebook  with

1) Only use 10,000 documents for training and test sets. 

2) Use the classification SVM. 

HW5: Clustering
In this homework, we use the Iris dataset for comparing different clustering algorithms:

 

I) Use K-means with following parameters:

n_clusters=3,
init='k-means++',
n_init=10,
max_iter=300,
tol=1e-04,
random_state=0, 

 a) Plot the clusters in 3-D plot with 3 features and 2-D plot with 2 features (similar to input     [6] of this  notebook, you can choose any combination that shows  clear clusters)

 b) Use elbow method to plot and choose the best "k", similar to inputs [7] and [8] of this notebook.

 c) Show the silhouette plots of these clusters similar to input  [9] of this  notebook. 

 d) Show the silhouette plots of K-means with n_clusters=2, 3, and 4.  
 

II) Performing the hierarchical clustering on a distance matrix, similar to inputs [14] to [19] of this notebook.

 
III) Performing the DBSCAN and compare it to K-means and agglomerative clustering, similar to inputs [24] to [25] of this notebook.


HW6 - Feedforward Neural Networks
I) Theory: Derive the following parts:

1. The backpropagation algorithm 

2. The derivative of sigmoid function

 

II) Coding: Following the content and codes of this chapter, the dataset is of course MNIST

  1. Compare the accuracy results between 20 and 40 hidden neuron network,  using the same 1,000 training images, cross-entropy cost function, learning rate of η=0.5, mini-batch size of 10,  and 300 epochs. 

  2. With a 40 hidden neuron network, compare the accuracy results between 1,000 and 5,000 training images, using cross-entropy cost function, learning rate of η=0.5, mini-batch size of 10,  and 300 epochs.

  3. With a 40 hidden neuron network, compare the accuracy results between with and without regularization, using the same 1,000 training images, cross-entropy cost function, learning rate of η=0.5, mini-batch size of 10,  and 300 epochs, λ is chosen in {0.1, 1, 10}

  4. Using heuristic approach, find the best hyper-parameters for learning a  40 hidden neuron network using the same 3,000 training images, cross-entropy cost function, mini-batch size of 10,  and 300 epochs. Show the accuracy of learning epochs with these chosen parameters. 
  
 HW7 - Deep Neural Network using TensorFlow or PyTorch
 

Perform the following tasks using either TensorFlow or PyTorch (Many reports that PyTorch is more intuitive)
  1. Compare the accuracy results between 30 and 50 hidden neuron network,  using the same 1,000 training images, cross-entropy cost function, learning rate of η=0.5, mini-batch size of 10,  and 300 epochs. 

  2. With a 40 hidden neuron network, compare the accuracy results between 1,000 and 5,000 training images, using cross-entropy cost function, learning rate of η=0.5, mini-batch size of 10,  and 300 epochs.

  3. With a 40 hidden neuron network, compare the accuracy results between L1 and L2 regularization, using the same 1,000 training images, cross-entropy cost function, learning rate of η=0.5, mini-batch size of 10,  and 300 epochs, λ is chosen in {0.1, 1, 10}

  4. Comparing different activation functions: ReLU, sigmoid, tanh, by using best hyper-parameters for learning a  40 hidden neuron network using the same 3,000 training images, cross-entropy cost function, mini-batch size of 10,  and 300 epochs. Show the accuracy of learning epochs of compared methods.
