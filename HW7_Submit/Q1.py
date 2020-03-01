import tensorflow as tf
import tensorflow.contrib.keras as keras
import numpy as np

# importing mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

epochsNo = 300
mini_batch_size = 10
eta = 0.5
trainingSample = 1000
testingSample = 100
noOfNeuron = 30

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

# makinf keras model
y_train_onehot = keras.utils.to_categorical(y_train)
model = keras.models.Sequential()
model.add(
    keras.layers.Dense(
        units=noOfNeuron,
        input_dim=X_train_centered.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units=noOfNeuron,
        input_dim=noOfNeuron,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units=y_train_onehot.shape[1],
        input_dim=noOfNeuron,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'))

# declare the optimizer and cost function
sgd_optimizer = keras.optimizers.SGD(lr=eta)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
history = model.fit(X_train_centered, y_train_onehot,
                    batch_size=mini_batch_size, epochs=epochsNo,
                    verbose=1,
                    validation_split=0.1)

# checking accuracy on training and testing dataset
y_train_pred = model.predict_classes(X_train_centered, verbose=0)

correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = model.predict_classes(X_test_centered, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))



epochsNo = 300
mini_batch_size = 10
eta = 0.5
trainingSample = 1000
testingSample = 100
noOfNeuron = 50

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

# makinf keras model
y_train_onehot = keras.utils.to_categorical(y_train)
model = keras.models.Sequential()
model.add(
    keras.layers.Dense(
        units=noOfNeuron,
        input_dim=X_train_centered.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units=noOfNeuron,
        input_dim=noOfNeuron,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units=y_train_onehot.shape[1],
        input_dim=noOfNeuron,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'))

# declare the optimizer and cost function
sgd_optimizer = keras.optimizers.SGD(lr=eta)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
history = model.fit(X_train_centered, y_train_onehot,
                    batch_size=mini_batch_size, epochs=epochsNo,
                    verbose=1,
                    validation_split=0.1)

# checking accuracy on training and testing dataset
y_train_pred = model.predict_classes(X_train_centered, verbose=0)

correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = model.predict_classes(X_test_centered, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))





