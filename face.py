import numpy as np
import time

def getFaceData(type: int):
    """
    Type:
        0 : Test
        1 : Train
        2 : Validation

    Returns:
        faces, labels
    """
    match type:
        case 0:
            value = "test"
        case 1:
            value = "train"
        case 2:
            value = "validation"
        case _:
            return
    with open(f"data/facedata/facedata{value}") as f:
        data = f.read().splitlines()
        faces = []
        for i in range(0, len(data), 70):
            section = data[i:i+70]
            face = np.array([[0 if pixel == " " else 1 for pixel in row] for row in section], dtype=np.int32)
            faces.append(face)
    with open(f"data/facedata/facedata{value}labels") as f:
        labels = [int(label) for label in f.read().splitlines()]
    return faces, labels

def getFeatures(x):
    features = []
    for row in range(0, 7):
        for col in range(0,6):
            range_data = x[row*10:(row+1)*10, col*10:(col+1)*10]
            features.append(np.count_nonzero(range_data == 1))
    return features

def trainPerceptron(data, labels):
    """
    This method computes trains for 10 minutes then returns the weights

    Returns:
        weights
    """
    weights = np.random.uniform(-1,1, 42+1)
    start_time = time.time()
    while True:
        # prev_weights[:] = weights
        for face_num in range(len(data)):
            features = getFeatures(data[face_num])
            output = 0
            for i in range(42):
                output += weights[i] * features[i]
            output += weights[42]
            if (output >= 0 and labels[face_num] == 0):
                for i in range(42):
                    weights[i] = weights[i] - features[i]
                weights[42] = weights[42] - 1
            elif (output < 0 and labels[face_num] == 1):
                for i in range(42):
                    weights[i] = weights[i] + features[i]
                weights[42] = weights[42] + 1
        if time.time() - start_time >= 600: # if 10 minutes has passed break out of loop
            break
        # if np.allclose(weights, prev_weights, atol=0.001):
        #     break
    return weights

def testPerceptron(data, labels, weights):
    totalCorrect = 0
    for face_num in range(len(data)):
        features = getFeatures(data[face_num])
        output = 0
        for i in range(42):
            output += weights[i] * features[i]
        output += weights[42]
        if ((output >= 0 and labels[face_num] == 1) or (output < 0 and labels[face_num] == 0)):
            totalCorrect +=1
    return totalCorrect/len(data)

def perceptron():
    faceDataTrain, faceDataTrainLabels = getFaceData(1)
    weights = trainPerceptron(faceDataTrain, faceDataTrainLabels)
    faceDataTest, faceDataTestLabels = getFaceData(0)
    # print(weights)
    accuracy = testPerceptron(faceDataTest, faceDataTestLabels, weights)
    print(f"Accuracy for Perceptron: {round((accuracy) * 100, 2)}%")

def testNeural(data, labels, weights):
    w_i_h, w_h_o, b_i_h, b_h_o = weights
    totalCorrect = 0
    for face_num in range(len(data)):
        features = getFeatures(data[face_num])
        features = np.array(features)
        label = np.array(labels[face_num])
        features.shape += (1,)
        h_pre = b_i_h + w_i_h @ features
        h = 1 / (1 + np.exp(-h_pre))
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
        if ((label == 1 and o >= 0.5) or (label == 0 and o < 0.5)):
            totalCorrect += 1
    return totalCorrect/len(data)

def trainNeural(data, labels):
    w_i_h = np.random.uniform(-1,1, (20,42))
    w_h_o = np.random.uniform(-1,1, (1,20))
    b_i_h = np.zeros((20, 1))
    b_h_o = np.zeros((1, 1))
    learn_rate = 0.01
    nr_correct = 0
    start_time = time.time()
    while True:
        for face_num in range(len(data)):
            features = getFeatures(data[face_num])
            features = np.array(features)
            label = np.array(labels[face_num])
            features.shape += (1,)

            h_pre = b_i_h + w_i_h @ features
            h = 1 / (1 + np.exp(-h_pre))

            o_pre = b_h_o + w_h_o @ h
            o = 1 / (1 + np.exp(-o_pre))

            e = 1 / len(o) * np.sum((o - label) ** 2, axis=0)
            if ((label == 1 and o >= 0.5) or (label == 0 and o < 0.5)):
                nr_correct += 1

            delta_o = o - label

            w_h_o += -learn_rate * delta_o @ np.transpose(h)
            b_h_o += -learn_rate * delta_o

            delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
            w_i_h += -learn_rate * delta_h @ np.transpose(features)
            b_i_h += -learn_rate * delta_h
        # print(f"{nr_correct / len(data)} {nr_correct} {len(data)}")
        nr_correct = 0
        if time.time() - start_time >= 600: # if 10 minutes has passed break out of loop
            break
    return w_i_h, w_h_o, b_i_h, b_h_o


def neural():
    faceDataTrain, faceDataTrainLabels = getFaceData(1)
    weights = trainNeural(faceDataTrain, faceDataTrainLabels)
    faceDataTrain, faceDataTrainLabels = getFaceData(0)
    # print(weights)
    accuracy = testNeural(faceDataTrain, faceDataTrainLabels, weights)
    print(f"Accuracy for Neural Network: {round((accuracy) * 100, 2)}%")

perceptron()
neural()