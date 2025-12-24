CS 430 – Homework 3, Problem 02
Author: Wyatt Laughner and Azaria A. Reed

This program implements binary logistic regression using SciPy’s BFGS optimizer to classify Iris-setosa (1) versus Iris-versicolor and Iris-virginica (0).

Files:
- main.py  – Python source code (logistic regression implementation)
- iris.data – dataset provided by the instructor
- sigmoid.py – helper module defining the sigmoid function

How to Run:
1. Install Python 3.x
2. Install NumPy and SciPy if not already installed:
   pip install numpy scipy
3. Place iris.data, sigmoid.py, and main.py in the same folder.
4. Run the program:
   python3 main.py

Program Description:
- Loads the Iris dataset and maps targets: Setosa → 1, others → 0.
- Performs a stratified 80/20 train–validation split (40 training and 10 validation samples per class).
- Normalizes training and validation features using the training statistics.
- Adds an intercept term to the feature matrix.
- Trains a binary classifier via scipy.optimize.minimize with the BFGS method, using custom cost and gradient functions.
- Evaluates model performance on the validation set by computing:
  - Optimal θ parameters
  - Confusion matrix (TP, FP, FN, TN)
  - Accuracy
  - Precision

Expected Output:
The console displays:
  Optimal Thetas: [ θ0, θ1, θ2, θ3, θ4 ]
  Confusion Matrix (TP, FP, FN, TN): [ …, …, …, … ]
  Accuracy:   xx.xxx%
  Precision:  xx.xxx%

Dependencies:
- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.6

Notes:
- This submission implements Problem 02 (CS 430) only—one classifier (1 vs 2 & 3) as specified in the assignment.
