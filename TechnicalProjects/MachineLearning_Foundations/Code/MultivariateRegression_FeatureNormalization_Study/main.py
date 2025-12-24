import sys
import numpy as np

ALPHA = 0.01
EPSILON = sys.float_info.epsilon

def GetData():
    fileName = "boston.txt"
    
    features = []
    targets = []
    file = open(fileName, "r")

    for line in file:
        row = line.strip().split()
        try:
            float(row[0])
        except:
            pass
        else:
            if len(row) == 11:
                features.append(row)
            if len(row) == 3:
                features[-1].append(row[0])
                features[-1].append(row[1])
                targets.append(row[2])
    
    file.close()
    return features, targets

def SplitData(features, targets):
    trainingFeatures = [[float(column) for column in row] for row in features[0:-50]]
    trainingTargets = [float(target) for target in targets[0:-50]]
    validationFeatures = [[float(column) for column in row] for row in features[-50:]]
    validationTargets = [float(target) for target in targets[-50:]]

    return  trainingFeatures,  trainingTargets, validationFeatures, validationTargets

def NormalizeData(trainingFeatures, validationFeatures):
    trainingMatrix = np.array(trainingFeatures)
    validationMatrix = np.array(validationFeatures)

    trainingFeatureMeans = trainingMatrix.mean(axis=0)
    trainingFeatureStandardDeviations = trainingMatrix.std(axis=0)
    trainingFeatureStandardDeviations[trainingFeatureStandardDeviations == 0.0] = 1.0

    normalizedTrainingFeatures = (trainingMatrix - trainingFeatureMeans) / trainingFeatureStandardDeviations
    normalizedValidationFeatures = (validationMatrix - trainingFeatureMeans) / trainingFeatureStandardDeviations

    return normalizedTrainingFeatures.tolist(), normalizedValidationFeatures.tolist()
    
def GradientDescent(examples, targets):
    m = len(examples)
    thetas = [0.0] * len(examples[0])
    
    previousCost = float("inf")
    difference = float("inf")
    while difference >= EPSILON:
        predictedTargets = []
        for example in examples:
            predictedTargets.append(sum([(thetas[i] * example[i]) for i in range(len(example))]))

        dJThetas = []
        for j in range(len(thetas)):
            dJThetas.append(sum([((predictedTargets[i] - targets[i]) * examples[i][j]) for i in range(m)]) / m)

        thetas = [(thetas[j] - (ALPHA * dJThetas[j])) for j in range(len(thetas))]

        cost = sum([((predictedTargets[i] - targets[i]) ** 2) for i in range(m)]) / (2 * m)
        difference = abs(previousCost - cost)
        previousCost = cost
        
    return thetas

def NormalEquation(features, targets): 
    X = np.array(features)
    y = np.array(targets)

    thetas = np.linalg.inv(X.T @ X) @ X.T @ y
    return thetas.tolist()

def WriteOutput(parameters2a, meanSquaredError2a, parameters2aNormalEquation, meanSquaredError2aNormalEquation, parameters2b, meanSquaredError2b):
    fileName = "output.txt"
    file = open(fileName, "w")

    file.write("Task 2a\n")
    file.write("Gradient Descent\n")
    file.write(f"Parameters: {parameters2a}\n")
    file.write(f"Mean Squared Error: {meanSquaredError2a}\n\n")

    file.write("Normal Equation\n")
    file.write(f"Parameters: {parameters2aNormalEquation}\n")
    file.write(f"Mean Squared Error: {meanSquaredError2aNormalEquation}\n\n")

    file.write("Task 2b\n")
    file.write("Gradient Descent\n")
    file.write(f"Parameters: {parameters2b}\n")
    file.write(f"Mean Squared Error: {meanSquaredError2b}\n")
    
    file.close()
    
def Main():
    features, targets = GetData()
    trainingFeatures, trainingTargets, validationFeatures, validationTargets = SplitData(features, targets)

    # Task 2a
    trainingFeatures2a = [[row[6], row[9]] for row in trainingFeatures]
    validationFeatures2a = [[row[6], row[9]] for row in validationFeatures]
    
    normalizedTrainingFeatures2a, normalizedValidationFeatures2a = NormalizeData(trainingFeatures2a, validationFeatures2a)
    normalizedTrainingFeatures2a = [[1.0] + row for row in normalizedTrainingFeatures2a]
    normalizedValidationFeatures2a = [[1.0] + row for row in normalizedValidationFeatures2a]
    
    parameters2a = GradientDescent(normalizedTrainingFeatures2a, trainingTargets)
    predictedValidationTargets2a = []
    for example in normalizedValidationFeatures2a:
        predictedValidationTargets2a.append(sum([(parameters2a[i] * example[i]) for i in range(len(example))]))
    meanSquaredError2a = sum([((predictedValidationTargets2a[i] - validationTargets[i]) ** 2) for i in range(len(validationTargets))]) / len(validationTargets)

    trainingFeatures2aWithIntercept = [[1.0] + row for row in trainingFeatures2a]
    validationFeatures2aWithIntercept = [[1.0] + row for row in validationFeatures2a]

    parameters2aNormalEquation = NormalEquation(trainingFeatures2aWithIntercept, trainingTargets)
    predictedValidationTargets2aNormalEquation = []
    for example in validationFeatures2aWithIntercept:
        predictedValidationTargets2aNormalEquation.append(sum([(parameters2aNormalEquation[i] * example[i]) for i in range(len(example))]))
    meanSquaredError2aNormalEquation = sum([((predictedValidationTargets2aNormalEquation[i] - validationTargets[i]) ** 2) for i in range(len(validationTargets))]) / len(validationTargets)

    # Task 2b
    normalizedTrainingFeatures2b, normalizedValidationFeatures2b = NormalizeData(trainingFeatures, validationFeatures)
    normalizedTrainingFeatures2b = [[1.0] + row for row in normalizedTrainingFeatures2b]
    normalizedValidationFeatures2b = [[1.0] + row for row in normalizedValidationFeatures2b]

    parameters2b = GradientDescent(normalizedTrainingFeatures2b, trainingTargets)
    predictedValidationTargets2b = []
    for example in normalizedValidationFeatures2b:
        predictedValidationTargets2b.append(sum([(parameters2b[i] * example[i]) for i in range(len(example))]))
    meanSquaredError2b = sum([((predictedValidationTargets2b[i] - validationTargets[i]) ** 2) for i in range(len(validationTargets))]) / len(validationTargets)

    # Output
    WriteOutput(parameters2a, meanSquaredError2a, parameters2aNormalEquation, meanSquaredError2aNormalEquation, parameters2b, meanSquaredError2b)

if __name__ == "__main__":
    Main()
