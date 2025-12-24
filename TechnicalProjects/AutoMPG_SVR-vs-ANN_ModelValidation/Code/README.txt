CS 430 – Term Project (Auto MPG Regression)
Authors: Wyatt Laughner and Azaria A. Reed

This program implements two regression models—Support Vector Regression (SVR) and Artificial Neural Network Regression (ANN)—to predict automobile fuel efficiency (MPG) using the Auto MPG dataset. The program trains both models, evaluates them using mean squared error (MSE), writes the results to a text file, and generates plots for SVR predictions and ANN convergence behavior.

Files:
- main.py                       – Python source code (data loading, preprocessing, model training, evaluation, and plotting)
- auto-mpg.data                 – Auto MPG dataset used by the program
- model_results_svr_ann_mse.txt – generated file containing model evaluation results (model name and MSE for each model)
- svr_predicted_vs_actual.png   – generated SVR predicted-versus-actual MPG scatter plot (created by main.py)
- ann_convergence_plot.png      – generated ANN training vs validation MSE plot (created by main.py)

How to Run:
1. Install Python 3.x.
2. Install the required packages:
       pip install numpy pandas scikit-learn matplotlib
3. Make sure auto-mpg.data and main.py are in the same directory.
4. Run the program from a terminal or command prompt:
       python3 main.py

Program Description:
- Loads the Auto MPG dataset from auto-mpg.data and assigns the appropriate column names.
- Treats "?" as missing data and drops any rows that contain missing values.
- Uses the following seven numeric features as inputs:
      cylinders, displacement, horsepower, weight, acceleration, model_year, origin
- Uses mpg as the target value to be predicted.
- Splits the data into training and testing sets using an 80/20 train–test split
  (scikit-learn train_test_split with a fixed random state for reproducibility).

Support Vector Regressor (SVR):
- Builds a scikit-learn Pipeline that:
      • Scales features with StandardScaler.
      • Trains an SVR model with an RBF kernel: SVR(kernel="rbf").
- Fits this pipeline on the training features and targets.
- Uses the trained SVR pipeline to predict MPG on the test set.
- Computes the test-set MSE using mean_squared_error(testTarget, predictedTarget).
- Opens model_results_svr_ann_mse.txt in write ("w") mode and writes:
      • The model name: "Support Vector Regressor"
      • The corresponding MSE (formatted to 5 decimal places).
- Creates a scatter plot of actual vs predicted MPG on the test set with:
      • x-axis: Actual MPG
      • y-axis: Predicted MPG
      • Title: "SVR Predicted vs Actual MPG"
  and saves this figure as:
      svr_predicted_vs_actual.png

Artificial Neural Network (ANN):
The ANN is handled in two parts:

1. Final ANN model for evaluation:
   - Builds a Pipeline with:
         • StandardScaler
         • MLPRegressor(
               hidden_layer_sizes=(64, 32),
               activation="relu",
               solver="adam",
               max_iter=1000,
               random_state=42
           )
   - Fits this ANN pipeline on the full training set, suppressing ConvergenceWarning
     using warnings.catch_warnings and filterwarnings.
   - This trained pipeline is used to make predictions on the test set.
   - Computes the test-set MSE using mean_squared_error(testTarget, predictedTarget).
   - Opens model_results_svr_ann_mse.txt in append ("a") mode and appends:
         • The model name: "Artificial Neural Network"
         • The corresponding MSE (formatted to 5 decimal places).

2. Curve-tracking ANN for convergence plotting:
   - Creates a separate StandardScaler and fits it on the full training features.
   - Transforms the training features to obtain scaledTrainFeaturesAll.
   - Splits the scaled training data into an inner training set and a validation set (80/20)
     using train_test_split with the same RANDOM_STATE.
   - Constructs a second MLPRegressor with:
         • hidden_layer_sizes=(64, 32)
         • activation="relu"
         • solver="adam"
         • max_iter=1
         • warm_start=True
         • random_state=42
   - Trains this curve-tracking ANN for NUM_EPOCHS (default 100) epochs in a loop:
         • Each epoch calls fit(...) once (one additional pass over the inner training data).
         • After each epoch, it:
               - Predicts on the inner training set and computes training MSE.
               - Predicts on the validation set and computes validation MSE.
               - Appends these values to trainErrors and validationErrors lists.
         • ConvergenceWarning is suppressed inside this loop similarly.
   - After training, uses trainErrors and validationErrors to generate a convergence plot:
         • x-axis: Epoch (1 to NUM_EPOCHS)
         • y-axis: Mean Squared Error
         • Two lines: "Training MSE" and "Validation MSE"
     and saves this figure as:
         ann_convergence_plot.png

Evaluation and Output:
- The program does not print model evaluation metrics to the console.
- All MSE results are written to:
      model_results_svr_ann_mse.txt
  in the following format:
      Support Vector Regressor
      Mean Squared Error: xx.xxxxx

      Artificial Neural Network
      Mean Squared Error: xx.xxxxx

- The following image files are saved in the same directory as main.py:
      svr_predicted_vs_actual.png
      ann_convergence_plot.png

Dependencies:
- Python ≥ 3.8
- NumPy
- Pandas
- scikit-learn
- Matplotlib

Notes:
- Feature scaling for SVR and the final ANN model is handled via scikit-learn Pipelines using StandardScaler.
- A separate StandardScaler is used for the internal curve-tracking ANN that generates convergence data.
- Convergence warnings from MLPRegressor are intentionally suppressed using warnings.catch_warnings and filterwarnings
  to keep the program output clean.
- This project satisfies the CS 430 term project requirements by:
      • Using regression models (SVR and ANN) to predict MPG.
      • Reporting test-set mean squared error.
      • Providing visualizations of model performance and ANN convergence behavior.
