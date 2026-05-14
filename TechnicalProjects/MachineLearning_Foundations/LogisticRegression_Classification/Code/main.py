from sigmoid import sigmoid
from scipy.optimize import minimize

import sys
import random
import numpy as np

ALPHA = 0.01
EPSILON = sys.float_info.epsilon

def GetData():
    fileName = "iris.data"

    features = []
    targets = []
    file = open(fileName, "r")

    for line in file:
        row = line.split(",")
        if len(row) == 5:
            features.append(row[0:4])
            if row[4].strip() == "Iris-setosa":
                targets.append(1)
            else:
                targets.append(0)

    file.close()
    return features, targets

def SplitData(features, targets):
    featuresClass1 = features[0:50]
    featuresClass2 = features[50:100]
    featuresClass3 = features[100:150]

    targetsClass1 = targets[0:50]
    targetsClass2 = targets[50:100]
    targetsClass3 = targets[100:150]

    trainingFeatures = []
    trainingTargets = []
    validationFeatures = []
    validationTargets = []

    SplitRandomly(featuresClass1, targetsClass1, trainingFeatures, trainingTargets, validationFeatures, validationTargets)
    SplitRandomly(featuresClass2, targetsClass2, trainingFeatures, trainingTargets, validationFeatures, validationTargets)
    SplitRandomly(featuresClass3, targetsClass3, trainingFeatures, trainingTargets, validationFeatures, validationTargets)

    return trainingFeatures, trainingTargets, validationFeatures, validationTargets

def SplitRandomly(unsplitFeatures, unsplitTargets, trainingFeatures, trainingTargets, validationFeatures, validationTargets):
    unsplit = []
    for i in range(50):
        unsplit.append(unsplitFeatures[i] + [unsplitTargets[i]])

    random.shuffle(unsplit)
    trainingSet = unsplit[0:40]
    validationSet = unsplit[40:50]

    for row in trainingSet:
        trainingFeatures.append([float(feature) for feature in row[0:4]])
        trainingTargets.append(row[4])

    for row in validationSet:
        validationFeatures.append([float(feature) for feature in row[0:4]])
        validationTargets.append(row[4])
        
def NormalizeData(trainingFeatures, validationFeatures):
    trainingMatrix = np.array(trainingFeatures)
    validationMatrix = np.array(validationFeatures)

    trainingFeatureMeans = trainingMatrix.mean(axis=0)
    trainingFeatureStandardDeviations = trainingMatrix.std(axis=0)
    trainingFeatureStandardDeviations[trainingFeatureStandardDeviations == 0.0] = 1.0

    normalizedTrainingFeatures = (trainingMatrix - trainingFeatureMeans) / trainingFeatureStandardDeviations
    normalizedValidationFeatures = (validationMatrix - trainingFeatureMeans) / trainingFeatureStandardDeviations

    return normalizedTrainingFeatures.tolist(), normalizedValidationFeatures.tolist()

def TrainBinaryClassifier(features, targets):
    features = np.array(features)
    targets = np.array(targets)

    thetas = np.zeros(features.shape[1])
    thetas = minimize(GetCost, thetas, args=(features, targets), jac=GetGradient, method="BFGS")

    return thetas.x.tolist()

def GetCost(thetas, features, targets):
    m = features.shape[0]
    cost = 0.0
    probabilities = features @ thetas
    for i in range(m):
        hypothesis = sigmoid(probabilities[i])
        if hypothesis < EPSILON:
            hypothesis = EPSILON
        elif hypothesis > 1.0 - EPSILON:
            hypothesis = 1.0 - EPSILON
        cost += (-targets[i] * np.log(hypothesis)) - ((1.0 - targets[i]) * np.log(1.0 - hypothesis))
    cost = cost / m

    return cost

def GetGradient(thetas, features, targets):
    features = np.array(features)
    targets = np.array(targets)
    hypotheses = sigmoid(features @ thetas)
    gradient = (features.T @ (hypotheses - targets)) / features.shape[0]

    return gradient

def ComputeValidationAccuracy(thetas, features, targets):
    features = np.array(features)
    targets = np.array(targets)

    confusionMatrix = [0] * 4
    probabilities = features @ thetas
    for i in range(features.shape[0]):
        hypothesis = sigmoid(probabilities[i])
        if hypothesis >= 0.5:
            hypothesis = 1.0
        else:
            hypothesis = 0.0

        if targets[i] == hypothesis and hypothesis == 1.0:
            confusionMatrix[0] += 1
        elif targets[i] == hypothesis and hypothesis == 0.0:
            confusionMatrix[3] += 1
        elif hypothesis == 1.0:
            confusionMatrix[1] += 1
        else:
            confusionMatrix[2] += 1

    return confusionMatrix

def ComputeAccuracy(confusionMatrix):
    accuracy = (confusionMatrix[0] + confusionMatrix[3]) / sum(confusionMatrix)
    return accuracy

def ComputePrecision(confusionMatrix):
    precision = confusionMatrix[0] / (confusionMatrix[0] + confusionMatrix[1])
    return precision
    
def Main():
    features, targets = GetData()
    trainingFeatures, trainingTargets, validationFeatures, validationTargets = SplitData(features, targets)

    normalizedTrainingFeatures, normalizedValidationFeatures = NormalizeData(trainingFeatures, validationFeatures)
    normalizedTrainingFeatures = [[1.0] + row for row in normalizedTrainingFeatures]
    normalizedValidationFeatures = [[1.0] + row for row in normalizedValidationFeatures]

    thetas = TrainBinaryClassifier(normalizedTrainingFeatures, trainingTargets)
    confusionMatrix = ComputeValidationAccuracy(thetas, normalizedValidationFeatures, validationTargets)
    accuracy = ComputeAccuracy(confusionMatrix)
    precision = ComputePrecision(confusionMatrix)

    print(f"Optimal Thetas: {thetas}")
    print(f"Confusion Matrix (TP, FP, FN, TN): {confusionMatrix}")
    print(f"Accuracy: {accuracy:.5%}")
    print(f"Precision: {precision:.5%}")

if __name__ == "__main__":
    Main()
