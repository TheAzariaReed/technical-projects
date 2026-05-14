import sys
import matplotlib.pyplot as plotter

ALPHA = 0.01
EPSILON = sys.float_info.epsilon

def GetData():
    fileName = "data.txt"
    
    populations = []
    profits = []
    file = open(fileName, "r")
    
    for line in file:
        trainingExample = line.strip().split(",")
        if len(trainingExample) == 2:
            populations.append(float(trainingExample[0]))
            profits.append(float(trainingExample[1]))
    
    file.close()
    return populations, profits
    
def PlotData(populations, profits):
    plotter.figure()
    plotter.scatter(populations, profits, marker="x", color="red", label="Training data")
    plotter.xlabel("Population of City in 10,000s")
    plotter.ylabel("Profit in $10,000s")
    plotter.title("Population vs. Profit")
    
def GradientDescent(populations, profits):
    m = len(populations)
    theta0 = 0.0
    theta1 = 0.0
    costs = []
    
    previousCost = float("inf")
    difference = float("inf")
    while difference >= EPSILON:
        predictedProfits = [(theta0 + (theta1 * population)) for population in populations]
        dJdTheta0 = sum([(predictedProfits[i] - profits[i]) for i in range(m)]) / m
        dJdTheta1 = sum([((predictedProfits[i] - profits[i]) * populations[i]) for i in range(m)]) / m
        theta0 -= (ALPHA * dJdTheta0)
        theta1 -= (ALPHA * dJdTheta1)
        cost = sum([((predictedProfits[i] - profits[i]) ** 2) for i in range(m)]) / (2 * m)
        costs.append(cost)
        difference = abs(previousCost - cost)
        previousCost = cost
        
    return theta0, theta1, costs

def PlotFitLine(populations, theta0, theta1):
    xCoordinates = [min(populations), max(populations)]
    yCoordinates = [(theta0 + (theta1 * xCoordinate)) for xCoordinate in xCoordinates]
    plotter.plot(xCoordinates, yCoordinates, color="blue", label="Linear regression")
    plotter.xlim(4, 24)
    plotter.xticks(list(range(4, 25, 2)))
    plotter.ylim(-5, 25)
    plotter.yticks(list(range(-5, 26, 5)))
    plotter.legend(loc="lower right")
    plotter.show()

def Main():
    populations, profits = GetData()
    PlotData(populations, profits)
    theta0, theta1, costs = GradientDescent(populations, profits)
    PlotFitLine(populations, theta0, theta1)
    prediction35 = theta0 + (theta1 * 3.5)
    prediction70 = theta0 + (theta1 * 7)
    print(f"theta0: {theta0:.5f}")
    print(f"theta1: {theta1:.5f}")
    print(f"Cost: {costs[-1]:.5f}")
    print(f"Prediction (35,000): {prediction35:.5f}")
    print(f"Prediction (70,000): {prediction70:.5f}")

if __name__ == "__main__":
    Main()
