CS 430 – Homework 1, Problem 02
Author: Azaria A. Reed

This program implements univariate linear regression with batch gradient descent.

Files:
- main.py (Python source code)
- data.txt (training data, instructor-provided)

How to run:
1. Install Python 3.x
2. Install matplotlib if not already installed:
   pip install matplotlib
3. Place data.txt in the same folder as main.py
4. Run:
   python3 main.py

Expected output:
- A plot showing training data (red x) and fitted regression line (blue).
- Console output with:
  θ0, θ1 (final parameters)
  Final cost J(θ)
  Predictions for populations of 35,000 and 70,000