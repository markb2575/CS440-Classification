import numpy as np
import time
import random

def getDigitsData(type: int):
    """
    Type:
        0 : Test
        1 : Training
        2 : Validation

    Returns:
        digits, labels
    """
    match type:
        case 0:
            value = "test"
        case 1:
            value = "training"
        case 2:
            value = "validation"
        case _:
            return
    with open(f"data/digitdata/{value}images") as f:
        data = f.read().splitlines()
        digits = []
        for i in range(0, len(data), 28):
            section = data[i:i+28]
            digit = np.array([[1 if pixel == "#" else 0.5 if pixel == "+" else 0 for pixel in row] for row in section], dtype=np.float32)
            digits.append(digit)
    with open(f"data/digitdata/{value}labels") as f:
        labels = [int(label) for label in f.read().splitlines()]
    return digits, labels

def getFeatures(x):
    features = []
    for row in range(0, 7):
        for col in range(0,7):
            range_data = x[row*4:(row+1)*4, col*4:(col+1)*4]
            edges = 0
            full = 0
            for section in range_data:
                for i in section:
                    if i == .5:
                        edges += 1
                    elif i == 1:
                        full += 1
            features.append((edges + full)/49)
    return features

def trainPerceptron(data, labels):
    """
    This method computes trains for 10 minutes then returns the weights

    Returns:
        weights
    """
    digitsDataTest,digitsDataTestLabels = getDigitsData(0)
    weights = [np.random.uniform(-1, 1, 49 + 1) for _ in range(10)]

    highestAccuracy = 0
    consecutiveBelow = 0
    while True:
        for digits_num in range(len(data)):
            outputs = []
            features = getFeatures(data[digits_num])
            for weight_num in range(len(weights)):
                output = 0
                for i in range(49):
                    output += weights[weight_num][i] * features[i]
                output += weights[weight_num][49]
                outputs.append(output)
            prediction = np.argmax(outputs)
            for i in range(49):
                weights[prediction][i] -= features[i]
            weights[prediction][49] -= 1
            for i in range(49):
                weights[labels[digits_num]][i] += features[i]
            weights[labels[digits_num]][49] += 1
        testResult = testPerceptron(digitsDataTest,digitsDataTestLabels,weights)
        # print(testResult)
        if testResult >= highestAccuracy:
            highestAccuracy = testResult
            consecutiveBelow = 0
        else:
            consecutiveBelow+=1
        if consecutiveBelow >= 5:
            break
        # if np.allclose(weights, prev_weights, atol=0.001):
        #     break
    return weights

def testPerceptron(data, labels, weights):
    totalCorrect = 0
    for digits_num in range(len(data)):
        features = getFeatures(data[digits_num])
        outputs = []
        for weight_num in range(len(weights)):
            output = 0
            for i in range(49):
                output += weights[weight_num][i] * features[i]
            output += weights[weight_num][49]
            outputs.append(output)
        prediction = np.argmax(outputs)
        if (prediction == labels[digits_num]):
            totalCorrect +=1
    return totalCorrect/len(data)

def testNeural(data, labels, weights):
    w_i_h, w_h_o, b_i_h, b_h_o = weights
    totalCorrect = 0
    for digits_num in range(len(data)):
        features = getFeatures(data[digits_num])
        features = np.array(features)
        label = np.array(labels[digits_num])
        features.shape += (1,)
        h_pre = b_i_h + w_i_h @ features
        
        h = 1 / (1 + np.exp(-h_pre))
        
        o_pre = b_h_o + w_h_o @ h
        
        o = 1 / (1 + np.exp(-o_pre))
        # print(o)
        # print(np.argmax(o))
        # print(label)
        # print()
        if (np.argmax(o) == label):
            totalCorrect += 1
    return totalCorrect/len(data)

def trainNeural(data, labels):
    w_i_h = np.random.uniform(-1,1, (30,49))
    w_h_o = np.random.uniform(-1,1, (10,30))
    b_i_h = np.zeros((30, 1))
    b_h_o = np.zeros((10, 1))
    learn_rate = 0.01
    highestAccuracy = 0
    consecutiveBelow = 0
    digitsDataTest, digitsDataTestLabels = getDigitsData(0)
    while True:
        for digits_num in range(len(data)):
            features = getFeatures(data[digits_num])
            features = np.array(features)
            label = np.array(labels[digits_num])
            label = np.eye(10)[label]
            label = np.array(label)
            label.shape += (1,)
            features.shape += (1,)
            # Forward propagation input -> hidden
            h_pre = b_i_h + w_i_h @ features
            h = 1 / (1 + np.exp(-h_pre))
            # Forward propagation hidden -> output
            o_pre = b_h_o + w_h_o @ h
            o = 1 / (1 + np.exp(-o_pre))

            # Backpropagation output -> hidden (cost function derivative)
            delta_o = o - label
            w_h_o += -learn_rate * delta_o @ np.transpose(h)
            b_h_o += -learn_rate * delta_o
            # Backpropagation hidden -> input (activation function derivative)
            delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
            w_i_h += -learn_rate * delta_h @ np.transpose(features)
            b_i_h += -learn_rate * delta_h
        weights = w_i_h, w_h_o, b_i_h, b_h_o
        testResult = testNeural(digitsDataTest,digitsDataTestLabels,weights)
        # print(testResult)
        if testResult >= highestAccuracy:
            highestAccuracy = testResult
            consecutiveBelow = 0
        else:
            consecutiveBelow+=1
        if consecutiveBelow >= 5:
            break
    return w_i_h, w_h_o, b_i_h, b_h_o

def splitDataPoints(data, labels):
    dataWithLabels = list(zip(data, labels))
    size = len(dataWithLabels)
    return (random.sample(dataWithLabels, round(size * .1)), random.sample(dataWithLabels, round(size * .2)), random.sample(dataWithLabels, round(size * .3)), random.sample(dataWithLabels, round(size * .4)), 
            random.sample(dataWithLabels, round(size * .5)), random.sample(dataWithLabels, round(size * .6)), random.sample(dataWithLabels, round(size * .7)), random.sample(dataWithLabels, round(size * .8)), 
            random.sample(dataWithLabels, round(size * .9)), dataWithLabels)

def neural(training):
    if training:
        digitsDataTrain, digitsDataTrainLabels = getDigitsData(1)
        splits = splitDataPoints(digitsDataTrain, digitsDataTrainLabels)
        weights = []
        percent = 10
        for split in splits:
            data, label = zip(*split)
            start = time.time()
            weight = trainNeural(data,label)
            print(f"Neural Network on Digits with {percent}% of data points took {time.time() - start} seconds to train")
            weights.append(weight)
            percent += 10
        return weights
    else:
        digitsDataTest, digitsDataTestLabels = getDigitsData(0)
        percent = 10
        while percent <= 100:
            w_i_h = np.load("weights/neural_digits/" + str(percent) + "%/w_i_h.npy")
            w_h_o = np.load("weights/neural_digits/" + str(percent) + "%/w_h_o.npy")
            b_i_h = np.load("weights/neural_digits/" + str(percent) + "%/b_i_h.npy")
            b_h_o = np.load("weights/neural_digits/" + str(percent) + "%/b_h_o.npy")
            weights = w_i_h, w_h_o, b_i_h, b_h_o
            accuracy = testNeural(digitsDataTest, digitsDataTestLabels, weights)
            print(f"Accuracy for Neural Network with {percent}% Training Data: {round((accuracy) * 100, 2)}%")
            percent += 10
        print()

def perceptron(training):
    if training:
        digitsDataTrain, digitsDataTrainLabels = getDigitsData(1)
        splits = splitDataPoints(digitsDataTrain, digitsDataTrainLabels)
        weights = []
        percent = 10
        for split in splits:
            data, label = zip(*split)
            start = time.time()
            weight = trainPerceptron(data,label)
            print(f"Perceptron on Digits with {percent}% of data points took {time.time() - start} seconds to train")
            weights.append(weight)
            percent += 10
        return weights
    else:
        digitsDataTest, digitsDataTestLabels = getDigitsData(0)
        percent = 10
        while percent <= 100:
            weights = []
            for i in range(10):
                weights.append(np.load("weights/perceptron_digits/" + str(percent) + "%/" + str(i) + ".npy"))
            accuracy = testPerceptron(digitsDataTest, digitsDataTestLabels, weights)
            print(f"Accuracy for Perceptron with {percent}% Training Data: {round((accuracy) * 100, 2)}%")
            percent += 10
        print()

def training():
    """
    Trains the neural network and perceptron on digits data and saves weights to file system
    """
    perceptronWeights = perceptron(training=True)
    neuralWeights = neural(training=True)
    percent = 10
    for weight in neuralWeights:
        w_i_h, w_h_o, b_i_h, b_h_o = weight
        np.save("weights/neural_digits/" + str(percent) + "%/w_i_h", w_i_h)
        np.save("weights/neural_digits/" + str(percent) + "%/w_h_o", w_h_o)
        np.save("weights/neural_digits/" + str(percent) + "%/b_i_h", b_i_h)
        np.save("weights/neural_digits/" + str(percent) + "%/b_h_o", b_h_o)
        percent += 10
    percent = 10
    for weight in perceptronWeights:
        for i in range(10):
            np.save("weights/perceptron_digits/" + str(percent) + "%/" + str(i), weight[i])
        percent += 10

def testing():
    """
    Tests the neural network and perceptron on digits data and reports accuracy
    """
    perceptron(training=False)
    neural(training=False)

def runTestPerceptron(num):
    """
    Runs perceptron trained on 100% of data for digits on a certain test case (num)
    """
    try:
        data, labels = getDigitsData(0)
        weights = []
        for i in range(10):
            weights.append(np.load("weights/perceptron_digits/100%/" + str(i) + ".npy"))
        features = getFeatures(data[num])
        outputs = []
        for weight_num in range(len(weights)):
            output = 0
            for i in range(49):
                output += weights[weight_num][i] * features[i]
            output += weights[weight_num][49]
            outputs.append(output)
        prediction = np.argmax(outputs)
        print(f"Predicted Label: {prediction} Actual Label: {labels[num]}")
    except:
        print(f"Could not run test with input: {num}")

def runTestNeural(num):
    """
    Runs neural network trained on 100% of data for digits on a certain test case (num)
    """
    try:
        data, labels = getDigitsData(0)
        w_i_h = np.load("weights/neural_digits/100%/w_i_h.npy")
        w_h_o = np.load("weights/neural_digits/100%/w_h_o.npy")
        b_i_h = np.load("weights/neural_digits/100%/b_i_h.npy")
        b_h_o = np.load("weights/neural_digits/100%/b_h_o.npy")
        features = getFeatures(data[num])
        features = np.array(features)
        label = np.array(labels[num])
        features.shape += (1,)
        h_pre = b_i_h + w_i_h @ features
        h = 1 / (1 + np.exp(-h_pre))
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
        print(f"Predicted Label: {np.argmax(o)} Actual Label: {labels[num]}")
    except:
        print(f"Could not run test with input: {num}")




