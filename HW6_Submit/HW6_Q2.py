import mnist_loader
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2
# For example, if the list was [2, 3, 1] then it would be a three-layer network, with the first layer
# containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.

Nuron = 40
net = network2.Network([784, Nuron, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

trainingSample1 = 1000
testSample1 = 100
# net.SGD(training_data[:1000], 30, 10, 10.0, lmbda = 1000.0, evaluation_data=validation_data[:100], monitor_evaluation_accuracy=True)
# epochs  30 mini_batch_size  10 eta 1.0
epochs = 300
mini_batch = 10
eta = 0.5
evaluation_cost_20, evaluation_accuracy_20, training_cost_20, training_accuracy_20 = net.SGD(training_data[:trainingSample1], epochs, mini_batch, eta, lmbda = 5.0, evaluation_data=validation_data[:testSample1],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# net.save( "filename.json")
print evaluation_cost_20
print evaluation_accuracy_20
print training_cost_20
print training_accuracy_20


# This is for 40
Nuron = 40
net = network2.Network([784, Nuron, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
trainingSample2 = 5000
testSample2 = 500
# epochs  30 mini_batch_size  10 eta 1.0
epochs = 300
mini_batch = 10
eta = 0.5
evaluation_cost_40, evaluation_accuracy_40, training_cost_40, training_accuracy_40 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, eta, lmbda = 5.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# net.save( "filename.json")
print evaluation_cost_40
print evaluation_accuracy_40
print training_cost_40
print training_accuracy_40

import numpy as np

npaTestAccuracy_20 = np.asarray(evaluation_accuracy_20, dtype=np.float32)
print "Test npa : ", (npaTestAccuracy_20/testSample1)*100

# npaTrainingAccuracy_20 = np.asarray(training_accuracy, dtype=np.float32)
# print "Training npa : ", (npaTrainingAccuracy_20/trainingSample)*100


npaTestAccuracy_40 = np.asarray(evaluation_accuracy_40, dtype=np.float32)
print "Test npa : ", (npaTestAccuracy_40/testSample2)*100

# npaTrainingAccuracy_40 = np.asarray(training_accuracy, dtype=np.float32)
# print "Training npa : ", (npaTrainingAccuracy_40/trainingSample)*100






t = np.arange(0, 110, 10)
# plt.rc('font',family='Comic Sans MS')
fig, ax = plt.subplots()
# ax.plot( (npaTestAccuracy_20/testSample1)*100, color='b', marker='^', ls='--', lw=2.0, label='Accuracy for 1000 Data')
# ax.plot((npaTestAccuracy_40/testSample2)*100, color='g', marker='d', ls='-.', lw=2.0, label='Accuracy for 5000 Data')

ax.plot( (npaTestAccuracy_20/testSample1)*100, color='r',  lw=2.0, label='Accuracy for 1000 Data')
ax.plot((npaTestAccuracy_40/testSample2)*100, color='g', lw=2.0, label='Accuracy for 5000 Data')

# plt.ylabel('Energy Consumption',fontname="Comic Sans MS", fontsize = 22)
plt.ylabel('Evaluation Accuracy', fontsize = 18)
plt.xlabel('Epochs', fontsize = 18)
# plt.title("Energy Loss Comparison")
plt.legend(loc='best',fontsize = 18)
ax.grid(True)
ticklines = ax.get_xticklines() + ax.get_yticklines()
gridlines = ax.get_xgridlines()
ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

for line in ticklines:
    line.set_linewidth(3)

for line in gridlines:
    line.set_linestyle('-')

for line in gridlines:
    line.set_linestyle('-')

for label in ticklabels:
    label.set_color('black')
    label.set_fontsize('large')

plt.show()


