import digits
import face
import numpy as np
import matplotlib.pyplot as plt


#Trains all models
# face.training()
# digits.training()

#Tests all models
_, _ = face.testing()
_, _ = digits.testing()

#Run specific test cases
# face.runTestNeural(15)
# face.runTestPerceptron(15)
# digits.runTestNeural(15)
# digits.runTestPerceptron(15)

#Calculating averages / standard deviation


"""
percep = [] #stores each trial
net = []    #stores each trial
for i in range(5):
    face.training()
    p_acc, n_acc = face.testing()
    percep.append(p_acc)
    net.append(n_acc)
percep_percent = [] #stores by percentage
net_percent = []
for i in range(10):
    percep_percent.append([x[i] for x in percep])
    net_percent.append([x[i] for x in net])

avg_p_acc = [np.mean(x) for x in percep_percent]
avg_n_acc = [np.mean(x) for x in net_percent]
std_p = [np.std(x) for x in percep_percent]
std_n = [np.std(x) for x in net_percent]

plt.plot(range(10, 110, 10), avg_p_acc, label='Perceptron Test')
plt.xlabel('Percentage of Training Data Used In Training')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Percentage of Training Data (Face Perceptron)')
plt.legend()
plt.show()

plt.plot(range(10, 110, 10), avg_n_acc, label='Neural Network Test')
plt.xlabel('Percentage of Training Data Used In Training')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Percentage of Training Data (Face Neural Network)')
plt.legend()
plt.show()

plt.plot(range(10, 110, 10), std_p, label='Perceptron Test')
plt.xlabel('Percentage of Training Data Used In Training')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation vs Percentage of Training Data (Face Perceptron)')
plt.legend()
plt.show()

plt.plot(range(10, 110, 10), std_n, label='Neural Network Test')
plt.xlabel('Percentage of Training Data Used In Training')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation vs Percentage of Training Data (Face Neural Network)')
plt.legend()
plt.show()

percep = [] #stores each trial
net = []    #stores each trial
p_acc, n_acc = digits.testing()
percep.append(p_acc)
net.append(n_acc)
for i in range(1):
    digits.training()
    p_acc, n_acc = digits.testing()
    percep.append(p_acc)
    net.append(n_acc)
percep_percent = [] #stores by percentage
net_percent = []
for i in range(10):
    percep_percent.append([x[i] for x in percep])
    net_percent.append([x[i] for x in net])

avg_p_acc = [np.mean(x) for x in percep_percent]
avg_n_acc = [np.mean(x) for x in net_percent]
std_p = [np.std(x) for x in percep_percent]
std_n = [np.std(x) for x in net_percent]

plt.plot(range(10, 110, 10), avg_p_acc, label='Perceptron Test')
plt.xlabel('Percentage of Training Data Used In Training')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Percentage of Training Data (Digits Perceptron)')
plt.legend()
plt.show()

plt.plot(range(10, 110, 10), avg_n_acc, label='Neural Network Test')
plt.xlabel('Percentage of Training Data Used In Training')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Percentage of Training Data (Digits Neural Network)')
plt.legend()
plt.show()

plt.plot(range(10, 110, 10), std_p, label='Perceptron Test')
plt.xlabel('Percentage of Training Data Used In Training')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation vs Percentage of Training Data (Digits Perceptron)')
plt.legend()
plt.show()

plt.plot(range(10, 110, 10), std_n, label='Neural Network Test')
plt.xlabel('Percentage of Training Data Used In Training')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation vs Percentage of Training Data (Digits Neural Network)')
plt.legend()
plt.show()
"""