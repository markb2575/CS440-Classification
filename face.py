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

def train(data, labels):
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

def test(data, labels, weights):
    totalCorrect = 0
    for face_num in range(len(data)):
        features = getFeatures(data[face_num])
        output = 0
        for i in range(42):
            output += weights[i] * features[i]
        output += weights[42]
        if (output >= 0 and labels[face_num] == 1):
            totalCorrect +=1
        elif (output < 0 and labels[face_num] == 0):
            totalCorrect +=1
    return totalCorrect/len(data)

def perceptron():
    faceDataTrain, faceDataTrainLabels = getFaceData(1)
    weights = train(faceDataTrain, faceDataTrainLabels)
    faceDataTest, faceDataTestLabels = getFaceData(0)
    print(weights)
    accuracy = test(faceDataTest, faceDataTestLabels, weights)
    print(accuracy)

perceptron()