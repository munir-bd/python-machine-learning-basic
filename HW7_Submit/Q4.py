import tensorflow as tf
import tensorflow.contrib.keras as keras
import numpy as np
import matplotlib.pyplot as plt

# importing mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

epochsNo = 300
mini_batch_size = 10
eta = 0.5
trainingSample = 3000
testingSample = 300
noOfNeuron = 40

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
X_train = mnist.train.images[:trainingSample]
y_train = mnist.train.labels[:trainingSample]
X_test = mnist.test.images[:testingSample]
y_test = mnist.test.labels[:testingSample]

print(len(X_train))
print(len(X_test))

## mean centering and normalization:
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val
del X_train, X_test

act_fn = ['relu', 'tanh', 'sigmoid']
color = ['b', 'r', 'g']

for ite in range(3):
    y_train_onehot = keras.utils.to_categorical(y_train)
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=40,
            input_dim=X_train_centered.shape[1],
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    model.add(
        keras.layers.Dense(
            units=40,
            input_dim=40,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    model.add(
        keras.layers.Dense(
            units=y_train_onehot.shape[1],
            input_dim=40,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    # declare the optimizer and cost function
    sgd_optimizer = keras.optimizers.SGD(lr=eta)
    model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
    history = model.fit(X_train_centered, y_train_onehot,
                        batch_size=mini_batch_size, epochs=epochsNo,
                        verbose=0,
                        validation_split=0.1)

    # checking accuracy on training and testing dataset
    y_train_pred = model.predict_classes(X_train_centered, verbose=0)
    correct_preds = np.sum(y_train == y_train_pred, axis=0)
    train_acc = correct_preds / y_train.shape[0]
    print('Training accuracy for eta = 0.5  %s is: %.2f%%' % (act_fn[ite], train_acc * 100))

    y_test_pred = model.predict_classes(X_test_centered, verbose=0)
    correct_preds = np.sum(y_test == y_test_pred, axis=0)
    test_acc = correct_preds / y_test.shape[0]
    print('Test accuracy for eta = 0.5 %s is: %.2f%%' % (act_fn[ite], test_acc * 100))



epochsNo = 3000
mini_batch_size = 10
eta = 1
trainingSample = 3000
testingSample = 300
noOfNeuron = 40

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
X_train = mnist.train.images[:trainingSample]
y_train = mnist.train.labels[:trainingSample]
X_test = mnist.test.images[:testingSample]
y_test = mnist.test.labels[:testingSample]

print(len(X_train))
print(len(X_test))

## mean centering and normalization:
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val
del X_train, X_test

act_fn = ['relu', 'tanh', 'sigmoid']
color = ['b', 'r', 'g']

for ite in range(3):
    y_train_onehot = keras.utils.to_categorical(y_train)
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=40,
            input_dim=X_train_centered.shape[1],
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    model.add(
        keras.layers.Dense(
            units=40,
            input_dim=40,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    model.add(
        keras.layers.Dense(
            units=y_train_onehot.shape[1],
            input_dim=40,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    # declare the optimizer and cost function
    sgd_optimizer = keras.optimizers.SGD(lr=eta)
    model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
    history = model.fit(X_train_centered, y_train_onehot,
                        batch_size=mini_batch_size, epochs=epochsNo,
                        verbose=0,
                        validation_split=0.1)

    # checking accuracy on training and testing dataset
    y_train_pred = model.predict_classes(X_train_centered, verbose=0)
    correct_preds = np.sum(y_train == y_train_pred, axis=0)
    train_acc = correct_preds / y_train.shape[0]
    print('Training accuracy for eta = 1  %s is: %.2f%%' % (act_fn[ite], train_acc * 100))

    y_test_pred = model.predict_classes(X_test_centered, verbose=0)
    correct_preds = np.sum(y_test == y_test_pred, axis=0)
    test_acc = correct_preds / y_test.shape[0]
    print('Test accuracy for eta = 1 %s is: %.2f%%' % (act_fn[ite], test_acc * 100))




epochsNo = 300
mini_batch_size = 10
eta = 0.25
trainingSample = 3000
testingSample = 300
noOfNeuron = 40

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
X_train = mnist.train.images[:trainingSample]
y_train = mnist.train.labels[:trainingSample]
X_test = mnist.test.images[:testingSample]
y_test = mnist.test.labels[:testingSample]

print(len(X_train))
print(len(X_test))

## mean centering and normalization:
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val
del X_train, X_test

act_fn = ['relu', 'tanh', 'sigmoid']
color = ['b', 'r', 'g']

for ite in range(3):
    y_train_onehot = keras.utils.to_categorical(y_train)
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=40,
            input_dim=X_train_centered.shape[1],
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    model.add(
        keras.layers.Dense(
            units=40,
            input_dim=40,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    model.add(
        keras.layers.Dense(
            units=y_train_onehot.shape[1],
            input_dim=40,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    # declare the optimizer and cost function
    sgd_optimizer = keras.optimizers.SGD(lr=eta)
    model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
    history = model.fit(X_train_centered, y_train_onehot,
                        batch_size=mini_batch_size, epochs=epochsNo,
                        verbose=0,
                        validation_split=0.1)

    # checking accuracy on training and testing dataset
    y_train_pred = model.predict_classes(X_train_centered, verbose=0)
    correct_preds = np.sum(y_train == y_train_pred, axis=0)
    train_acc = correct_preds / y_train.shape[0]
    print('Training accuracy for eta = 0.25  %s is: %.2f%%' % (act_fn[ite], train_acc * 100))

    y_test_pred = model.predict_classes(X_test_centered, verbose=0)
    correct_preds = np.sum(y_test == y_test_pred, axis=0)
    test_acc = correct_preds / y_test.shape[0]
    print('Test accuracy for eta = 0.25 %s is: %.2f%%' % (act_fn[ite], test_acc * 100))



epochsNo = 300
mini_batch_size = 10
eta = 1.5
trainingSample = 3000
testingSample = 300
noOfNeuron = 40

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
X_train = mnist.train.images[:trainingSample]
y_train = mnist.train.labels[:trainingSample]
X_test = mnist.test.images[:testingSample]
y_test = mnist.test.labels[:testingSample]

print(len(X_train))
print(len(X_test))

## mean centering and normalization:
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val
del X_train, X_test

act_fn = ['relu', 'tanh', 'sigmoid']
color = ['b', 'r', 'g']

for ite in range(3):
    y_train_onehot = keras.utils.to_categorical(y_train)
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=40,
            input_dim=X_train_centered.shape[1],
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    model.add(
        keras.layers.Dense(
            units=40,
            input_dim=40,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    model.add(
        keras.layers.Dense(
            units=y_train_onehot.shape[1],
            input_dim=40,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=act_fn[ite]))
    # declare the optimizer and cost function
    sgd_optimizer = keras.optimizers.SGD(lr=eta)
    model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
    history = model.fit(X_train_centered, y_train_onehot,
                        batch_size=mini_batch_size, epochs=epochsNo,
                        verbose=0,
                        validation_split=0.1)

    # checking accuracy on training and testing dataset
    y_train_pred = model.predict_classes(X_train_centered, verbose=0)
    correct_preds = np.sum(y_train == y_train_pred, axis=0)
    train_acc = correct_preds / y_train.shape[0]
    print('Training accuracy for eta = 1.5  %s is: %.2f%%' % (act_fn[ite], train_acc * 100))

    y_test_pred = model.predict_classes(X_test_centered, verbose=0)
    correct_preds = np.sum(y_test == y_test_pred, axis=0)
    test_acc = correct_preds / y_test.shape[0]
    print('Test accuracy for eta = 1.5 %s is: %.2f%%' % (act_fn[ite], test_acc * 100))