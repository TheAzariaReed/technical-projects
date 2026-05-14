CS 430 – Homework 2, Problem 01
Author: Azaria A. Reed

This program implements multivariate linear regression on the Boston Housing dataset using both batch gradient descent and the normal equation.

Files:
- main.py (Python source code)
- boston.txt (dataset, instructor-provided)

How to run:
1. Install Python 3.x
2. Ensure NumPy is installed (if not, run):
   pip install numpy
3. Place boston.txt in the same folder as main.py
4. Run the program:
   python3 main.py

Expected output:
The console will display:
- Learned parameters (θ values) and Mean Squared Error (MSE) for:
  - Task 2a – Gradient Descent using AGE and TAX
  - Task 2a – Normal Equation (using AGE and TAX)
  - Task 2b – Gradient Descent using all features

The MSE values should approximate:
- Task 2a (Gradient Descent): ≈ 22.05
- Task 2a (Normal Equation): ≈ 22.05
- Task 2b (Gradient Descent): ≈ 10.96
