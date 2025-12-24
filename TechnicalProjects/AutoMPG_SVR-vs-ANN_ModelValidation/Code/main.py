import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

# Define the file name for the Auto MPG dataset.
DATA_FILE_NAME = "auto-mpg.data"

# Define the column names for the Auto MPG dataset.
COLUMN_NAMES = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name"
]

# Define the test size to create an 80-20 train-test split.
TEST_SIZE = 0.2

# Define the random state to keep results reproducible.
RANDOM_STATE = 42

# Define the number of epochs to train the ANN.
NUM_EPOCHS = 100

def LoadData():
    # Load the dataset from a local file and assign column names.
    dataFrame = pd.read_csv(
        DATA_FILE_NAME,
        sep=r"\s+",
        names=COLUMN_NAMES,
        na_values="?"
    )

    # Drop rows that contain missing values.
    dataFrame = dataFrame.dropna()

    # Select the feature columns that will be used for prediction.
    featureColumns = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin"
    ]

    # Separate the features and the target variable.
    features = dataFrame[featureColumns]
    target = dataFrame["mpg"]

    # Return the features and the target.
    return features, target

def SplitData(features, target):
    # Mirroring the homeworks, perform an 80-20 train-test split.
    trainFeatures, testFeatures, trainTarget, testTarget = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Return the training and testing features and targets.
    return trainFeatures, testFeatures, trainTarget, testTarget

def TrainSvrModel(trainFeatures, trainTarget):
    # Create a pipeline that scales features and trains an SVR model.
    svrPipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf"))
    ])

    # Fit the SVR pipeline on the training data.
    svrPipeline.fit(trainFeatures, trainTarget)

    # Return the trained SVR pipeline.
    return svrPipeline

def TrainAnnModel(trainFeatures, trainTarget):
    # Create a pipeline that scales features and trains an ANN model.
    annPipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])

    # Fit the ANN pipeline on the training data, suppressing convergence warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        annPipeline.fit(trainFeatures, trainTarget)
        
    # Create a scaler and scale all training features for curve tracking.
    scaler = StandardScaler()
    scaledTrainFeaturesAll = scaler.fit_transform(trainFeatures)

    # Split the scaled training data into inner training and validation sets for curves.
    innerTrainFeatures, validationFeatures, innerTrainTarget, validationTarget = train_test_split(
        scaledTrainFeaturesAll,
        trainTarget,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # Create an MLPRegressor that uses the Adam optimizer for curve tracking.
    curveRegressor = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_STATE
    )

    # Create lists to store the training and validation mean squared errors.
    trainErrors = []
    validationErrors = []

    # Train the curve-tracking ANN for a fixed number of epochs and record the errors.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for epochIndex in range(NUM_EPOCHS):
            # Fit the curve-tracking ANN for one epoch on the inner training data.
            curveRegressor.fit(innerTrainFeatures, innerTrainTarget)

            # Predict on the inner training set.
            trainPredictions = curveRegressor.predict(innerTrainFeatures)
            # Compute the training mean squared error.
            trainMse = mean_squared_error(innerTrainTarget, trainPredictions)
            # Append the training error to the list.
            trainErrors.append(trainMse)

            # Predict on the validation set.
            validationPredictions = curveRegressor.predict(validationFeatures)
            # Compute the validation mean squared error.
            validationMse = mean_squared_error(validationTarget, validationPredictions)
            # Append the validation error to the list.
            validationErrors.append(validationMse)

    # Return the trained ANN pipeline and the recorded errors for plotting.
    return annPipeline, trainErrors, validationErrors

def EvaluateModel(modelName, trainedModel, testFeatures, testTarget):
    # Evaluate the Support Vector Regressor model.
    if modelName == "Support Vector Regressor":
        # Use the trained SVR model to predict target values for the test features.
        predictedTarget = trainedModel.predict(testFeatures)

        # Compute the mean squared error.
        mse = mean_squared_error(testTarget, predictedTarget)

        # Write the SVR results to the output file.
        with open("model_results_svr_ann_mse.txt", "w") as file:
            file.write(f"{modelName}\n")
            file.write(f"Mean Squared Error: {mse:.5f}\n\n")

        # Create a predicted versus actual scatter plot for SVR.
        plt.figure(figsize=(6, 4))
        plt.scatter(testTarget, predictedTarget, alpha=0.7)
        plt.xlabel("Actual MPG")
        plt.ylabel("Predicted MPG")
        plt.title("SVR Predicted vs Actual MPG")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("svr_predicted_vs_actual.png")
        plt.close()

    # Evaluate the Artificial Neural Network model.
    else:
        # Unpack the ANN pipeline and the recorded errors.
        annPipeline, trainErrors, validationErrors = trainedModel

        # Use the trained ANN pipeline to predict target values for the test features.
        predictedTarget = annPipeline.predict(testFeatures)

        # Compute the mean squared error.
        mse = mean_squared_error(testTarget, predictedTarget)

        # Write the ANN results to the output file.
        with open("model_results_svr_ann_mse.txt", "a") as file:
            file.write(f"{modelName}\n")
            file.write(f"Mean Squared Error: {mse:.5f}\n\n")

        # Create a convergence plot for training and validation errors.
        epochIndices = range(1, len(trainErrors) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochIndices, trainErrors, label="Training MSE")
        plt.plot(epochIndices, validationErrors, label="Validation MSE")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.title("ANN Training vs Validation MSE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ann_convergence_plot.png")
        plt.close()

def Main():
    # Load and clean the dataset.
    features, target = LoadData()

    # Split the data into training and testing sets.
    trainFeatures, testFeatures, trainTarget, testTarget = SplitData(features, target)

    # Train the Support Vector Regressor model.
    svrModel = TrainSvrModel(trainFeatures, trainTarget)

    # Train the Artificial Neural Network model.
    annModel = TrainAnnModel(trainFeatures, trainTarget)

    # Evaluate the Support Vector Regressor model.
    EvaluateModel("Support Vector Regressor", svrModel, testFeatures, testTarget)

    # Evaluate the Artificial Neural Network model.
    EvaluateModel("Artificial Neural Network", annModel, testFeatures, testTarget)

if __name__ == "__main__":
    # Run the main function when the script is executed directly.
    Main()
