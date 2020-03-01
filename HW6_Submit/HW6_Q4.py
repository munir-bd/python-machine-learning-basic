import mnist_loader
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2
# For example, if the list was [2, 3, 1] then it would be a three-layer network, with the first layer
# containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.


#
# # This is for 40
# Nuron = 40
# net = network2.Network([784, Nuron, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# trainingSample2 = 3000
# testSample2 = 300
# # epochs  30 mini_batch_size  10 eta 1.0
# epochs = 300
# mini_batch = 10
# eta = 0.5
# evaluation_cost_01, evaluation_accuracy_01, training_cost_01, training_accuracy_01 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, eta, lmbda = 1.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# # net.save( "filename.json")
# print evaluation_cost_01
# print evaluation_accuracy_01
# print training_cost_01
# print training_accuracy_01
#
#
# evaluation_cost_1, evaluation_accuracy_1, training_cost_1, training_accuracy_1 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, eta, lmbda = 3.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# # net.save( "filename.json")
# print evaluation_cost_1
# print evaluation_accuracy_1
# print training_cost_1
# print training_accuracy_1
#
#
# evaluation_cost_10, evaluation_accuracy_10, training_cost_10, training_accuracy_10 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, eta, lmbda = 5.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# # net.save( "filename.json")
# print evaluation_cost_10
# print evaluation_accuracy_10
# print training_cost_10
# print training_accuracy_10
#
#
# import numpy as np
#
# npaTestAccuracy_01 = np.asarray(evaluation_accuracy_01, dtype=np.float32)
# print "Test npa : ", (npaTestAccuracy_01/testSample2)*100
#
# npaTestAccuracy_1 = np.asarray(evaluation_accuracy_1, dtype=np.float32)
# print "Test npa : ", (npaTestAccuracy_1/testSample2)*100
#
# npaTestAccuracy_10 = np.asarray(evaluation_accuracy_10, dtype=np.float32)
# print "Test npa : ", (npaTestAccuracy_10/testSample2)*100
#
# # npaTrainingAccuracy_40 = np.asarray(training_accuracy, dtype=np.float32)
# # print "Training npa : ", (npaTrainingAccuracy_40/trainingSample)*100
#
#
# t = np.arange(0, 110, 10)
# # plt.rc('font',family='Comic Sans MS')
# fig, ax = plt.subplots()
# # ax.plot( (npaTestAccuracy_01/testSample2)*100, color='b', marker='^', ls='--', lw=2.0, label='Test Accuracy Lamda 0.1')
# # ax.plot( (npaTestAccuracy_1/testSample2)*100, color='r', marker='o', ls='--', lw=2.0, label='Test Accuracy Lamda 1.0')
# # ax.plot((npaTestAccuracy_10/testSample2)*100, color='g', marker='d', ls='-.', lw=2.0, label='Test Accuracy Lamda 10.0')
#
# ax.plot( (npaTestAccuracy_01/testSample2)*100, color='b',  lw=2.0, label='Test Accuracy Lamda 1.0 & eta = 0.5')
# ax.plot( (npaTestAccuracy_1/testSample2)*100, color='r', lw=2.0, label='Test Accuracy Lamda 3.0 & eta = 0.5')
# ax.plot((npaTestAccuracy_10/testSample2)*100, color='g',  lw=2.0, label='Test Accuracy Lamda 5.0 & eta = 0.5')
#
# # plt.ylabel('Energy Consumption',fontname="Comic Sans MS", fontsize = 22)
# plt.ylabel('Evaluation Accuracy', fontsize = 18)
# plt.xlabel('Epochs', fontsize = 18)
# # plt.title("Energy Loss Comparison")
# plt.legend(loc='best',fontsize = 18)
# ax.grid(True)
# ticklines = ax.get_xticklines() + ax.get_yticklines()
# gridlines = ax.get_xgridlines()
# ticklabels = ax.get_xticklabels() + ax.get_yticklabels()
#
# for line in ticklines:
#     line.set_linewidth(3)
#
# for line in gridlines:
#     line.set_linestyle('-')
#
# for line in gridlines:
#     line.set_linestyle('-')
#
# for label in ticklabels:
#     label.set_color('black')
#     label.set_fontsize('large')
#
# plt.show()
#





#
# # This is for 40
# Nuron = 40
# net = network2.Network([784, Nuron, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# trainingSample2 = 3000
# testSample2 = 300
# # epochs  30 mini_batch_size  10 eta 1.0
# epochs = 300
# mini_batch = 10
# eta = 0.5
# evaluation_cost_01, evaluation_accuracy_01, training_cost_01, training_accuracy_01 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, 1.0, lmbda = 1.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# # net.save( "filename.json")
# print evaluation_cost_01
# print evaluation_accuracy_01
# print training_cost_01
# print training_accuracy_01
#
#
# evaluation_cost_1, evaluation_accuracy_1, training_cost_1, training_accuracy_1 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, 5.0, lmbda = 1.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# # net.save( "filename.json")
# print evaluation_cost_1
# print evaluation_accuracy_1
# print training_cost_1
# print training_accuracy_1
#
#
# evaluation_cost_10, evaluation_accuracy_10, training_cost_10, training_accuracy_10 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, 10.0, lmbda = 1.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# # net.save( "filename.json")
# print evaluation_cost_10
# print evaluation_accuracy_10
# print training_cost_10
# print training_accuracy_10
#
#
# import numpy as np
#
# npaTestAccuracy_01 = np.asarray(evaluation_accuracy_01, dtype=np.float32)
# print "Test npa : ", (npaTestAccuracy_01/testSample2)*100
#
# npaTestAccuracy_1 = np.asarray(evaluation_accuracy_1, dtype=np.float32)
# print "Test npa : ", (npaTestAccuracy_1/testSample2)*100
#
# npaTestAccuracy_10 = np.asarray(evaluation_accuracy_10, dtype=np.float32)
# print "Test npa : ", (npaTestAccuracy_10/testSample2)*100
#
# # npaTrainingAccuracy_40 = np.asarray(training_accuracy, dtype=np.float32)
# # print "Training npa : ", (npaTrainingAccuracy_40/trainingSample)*100
#
#
# t = np.arange(0, 110, 10)
# # plt.rc('font',family='Comic Sans MS')
# fig, ax = plt.subplots()
# # ax.plot( (npaTestAccuracy_01/testSample2)*100, color='b', marker='^', ls='--', lw=2.0, label='Test Accuracy Lamda 0.1')
# # ax.plot( (npaTestAccuracy_1/testSample2)*100, color='r', marker='o', ls='--', lw=2.0, label='Test Accuracy Lamda 1.0')
# # ax.plot((npaTestAccuracy_10/testSample2)*100, color='g', marker='d', ls='-.', lw=2.0, label='Test Accuracy Lamda 10.0')
#
# ax.plot( (npaTestAccuracy_01/testSample2)*100, color='b',  lw=2.0, label='Test Accuracy Lamda 1.0 & eta = 1.0')
# ax.plot( (npaTestAccuracy_1/testSample2)*100, color='r', lw=2.0, label='Test Accuracy Lamda 1.0 & eta = 5.0')
# ax.plot((npaTestAccuracy_10/testSample2)*100, color='g',  lw=2.0, label='Test Accuracy Lamda 1.0 & eta = 10.0')
#
# # plt.ylabel('Energy Consumption',fontname="Comic Sans MS", fontsize = 22)
# plt.ylabel('Evaluation Accuracy', fontsize = 18)
# plt.xlabel('Epochs', fontsize = 18)
# # plt.title("Energy Loss Comparison")
# plt.legend(loc='best',fontsize = 18)
# ax.grid(True)
# ticklines = ax.get_xticklines() + ax.get_yticklines()
# gridlines = ax.get_xgridlines()
# ticklabels = ax.get_xticklabels() + ax.get_yticklabels()
#
# for line in ticklines:
#     line.set_linewidth(3)
#
# for line in gridlines:
#     line.set_linestyle('-')
#
# for line in gridlines:
#     line.set_linestyle('-')
#
# for label in ticklabels:
#     label.set_color('black')
#     label.set_fontsize('large')
#
# plt.show()
#
#


# # This is for 40
# Nuron = 40
# net = network2.Network([784, Nuron, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# trainingSample2 = 3000
# testSample2 = 300
# # epochs  30 mini_batch_size  10 eta 1.0
# epochs = 300
# mini_batch = 10
# eta = 0.5
# evaluation_cost_01, evaluation_accuracy_01, training_cost_01, training_accuracy_01 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, 0.025, lmbda = 1.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# # net.save( "filename.json")
# print evaluation_cost_01
# print evaluation_accuracy_01
# print training_cost_01
# print training_accuracy_01
#
#
# evaluation_cost_1, evaluation_accuracy_1, training_cost_1, training_accuracy_1 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, 0.25, lmbda = 1.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# # net.save( "filename.json")
# print evaluation_cost_1
# print evaluation_accuracy_1
# print training_cost_1
# print training_accuracy_1
#
#
# evaluation_cost_10, evaluation_accuracy_10, training_cost_10, training_accuracy_10 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, 0.5, lmbda = 1.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# # net.save( "filename.json")
# print evaluation_cost_10
# print evaluation_accuracy_10
# print training_cost_10
# print training_accuracy_10
#
# evaluation_cost_11, evaluation_accuracy_11, training_cost_11, training_accuracy_11 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, 0.75, lmbda = 1.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# # net.save( "filename.json")
# print evaluation_cost_11
# print evaluation_accuracy_11
# print training_cost_11
# print training_accuracy_11
#
#
# import numpy as np
#
# npaTestAccuracy_01 = np.asarray(evaluation_accuracy_01, dtype=np.float32)
# print "Test npa : ", (npaTestAccuracy_01/testSample2)*100
#
# npaTestAccuracy_1 = np.asarray(evaluation_accuracy_1, dtype=np.float32)
# print "Test npa : ", (npaTestAccuracy_1/testSample2)*100
#
# npaTestAccuracy_10 = np.asarray(evaluation_accuracy_10, dtype=np.float32)
# print "Test npa : ", (npaTestAccuracy_10/testSample2)*100
#
# npaTestAccuracy_11 = np.asarray(evaluation_accuracy_11, dtype=np.float32)
# print "Test npa : ", (npaTestAccuracy_11/testSample2)*100
#
# # npaTrainingAccuracy_40 = np.asarray(training_accuracy, dtype=np.float32)
# # print "Training npa : ", (npaTrainingAccuracy_40/trainingSample)*100
#
#
# t = np.arange(0, 110, 10)
# # plt.rc('font',family='Comic Sans MS')
# fig, ax = plt.subplots()
# # ax.plot( (npaTestAccuracy_01/testSample2)*100, color='b', marker='^', ls='--', lw=2.0, label='Test Accuracy Lamda 0.1')
# # ax.plot( (npaTestAccuracy_1/testSample2)*100, color='r', marker='o', ls='--', lw=2.0, label='Test Accuracy Lamda 1.0')
# # ax.plot((npaTestAccuracy_10/testSample2)*100, color='g', marker='d', ls='-.', lw=2.0, label='Test Accuracy Lamda 10.0')
#
# ax.plot( (npaTestAccuracy_01/testSample2)*100, color='b',  lw=2.0, label='Test Accuracy Lamda 1.0 & eta = 0.025')
# ax.plot( (npaTestAccuracy_1/testSample2)*100, color='r', lw=2.0, label='Test Accuracy Lamda 1.0 & eta = 0.25')
# ax.plot((npaTestAccuracy_10/testSample2)*100, color='g',  lw=2.0, label='Test Accuracy Lamda 1.0 & eta = 0.5')
# ax.plot((npaTestAccuracy_11/testSample2)*100, color='y',  lw=2.0, label='Test Accuracy Lamda 1.0 & eta = 0.75')
#
#
# # plt.ylabel('Energy Consumption',fontname="Comic Sans MS", fontsize = 22)
# plt.ylabel('Evaluation Accuracy', fontsize = 18)
# plt.xlabel('Epochs', fontsize = 18)
# # plt.title("Energy Loss Comparison")
# plt.legend(loc='best',fontsize = 18)
# ax.grid(True)
# ticklines = ax.get_xticklines() + ax.get_yticklines()
# gridlines = ax.get_xgridlines()
# ticklabels = ax.get_xticklabels() + ax.get_yticklabels()
#
# for line in ticklines:
#     line.set_linewidth(3)
#
# for line in gridlines:
#     line.set_linestyle('-')
#
# for line in gridlines:
#     line.set_linestyle('-')
#
# for label in ticklabels:
#     label.set_color('black')
#     label.set_fontsize('large')
#
# plt.show()






# This is for 40
Nuron = 40
net = network2.Network([784, Nuron, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
trainingSample2 = 3000
testSample2 = 300
# epochs  30 mini_batch_size  10 eta 1.0
epochs = 300
mini_batch = 10
eta = 0.5
# evaluation_cost_01, evaluation_accuracy_01, training_cost_01, training_accuracy_01 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, 0.75, lmbda = 1.5, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# # net.save( "filename.json")
# print evaluation_cost_01
# print evaluation_accuracy_01
# print training_cost_01
# print training_accuracy_01


evaluation_cost_1, evaluation_accuracy_1, training_cost_1, training_accuracy_1 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, 0.5, lmbda = 1.0, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# net.save( "filename.json")
print evaluation_cost_1
print evaluation_accuracy_1
print training_cost_1
print training_accuracy_1


evaluation_cost_10, evaluation_accuracy_10, training_cost_10, training_accuracy_10 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, 0.75, lmbda = 0.75, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# net.save( "filename.json")
print evaluation_cost_10
print evaluation_accuracy_10
print training_cost_10
print training_accuracy_10

evaluation_cost_11, evaluation_accuracy_11, training_cost_11, training_accuracy_11 = net.SGD(training_data[:trainingSample2], epochs, mini_batch, 0.5, lmbda = 0.75, evaluation_data=validation_data[:testSample2],  monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True,monitor_training_accuracy= True)
# net.save( "filename.json")
print evaluation_cost_11
print evaluation_accuracy_11
print training_cost_11
print training_accuracy_11


import numpy as np

# npaTestAccuracy_01 = np.asarray(evaluation_accuracy_01, dtype=np.float32)
# print "Test npa : ", (npaTestAccuracy_01/testSample2)*100

npaTestAccuracy_1 = np.asarray(evaluation_accuracy_1, dtype=np.float32)
print "Test npa : ", (npaTestAccuracy_1/testSample2)*100

npaTestAccuracy_10 = np.asarray(evaluation_accuracy_10, dtype=np.float32)
print "Test npa : ", (npaTestAccuracy_10/testSample2)*100

npaTestAccuracy_11 = np.asarray(evaluation_accuracy_11, dtype=np.float32)
print "Test npa : ", (npaTestAccuracy_11/testSample2)*100

# npaTrainingAccuracy_40 = np.asarray(training_accuracy, dtype=np.float32)
# print "Training npa : ", (npaTrainingAccuracy_40/trainingSample)*100


t = np.arange(0, 110, 10)
# plt.rc('font',family='Comic Sans MS')
fig, ax = plt.subplots()
# ax.plot( (npaTestAccuracy_01/testSample2)*100, color='b', marker='^', ls='--', lw=2.0, label='Test Accuracy Lamda 0.1')
# ax.plot( (npaTestAccuracy_1/testSample2)*100, color='r', marker='o', ls='--', lw=2.0, label='Test Accuracy Lamda 1.0')
# ax.plot((npaTestAccuracy_10/testSample2)*100, color='g', marker='d', ls='-.', lw=2.0, label='Test Accuracy Lamda 10.0')

# ax.plot( (npaTestAccuracy_01/testSample2)*100, color='b',  lw=2.0, label='Test Accuracy Lamda 1.5 & eta = 0.75')
ax.plot( (npaTestAccuracy_1/testSample2)*100, color='r', lw=2.0, label='Test Accuracy Lamda 1.0 & eta = 0.5')
ax.plot((npaTestAccuracy_10/testSample2)*100, color='g',  lw=2.0, label='Test Accuracy Lamda 0.75 & eta = 0.75')
ax.plot((npaTestAccuracy_11/testSample2)*100, color='y',  lw=2.0, label='Test Accuracy Lamda 0.75 & eta = 0.5')


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



