import numpy as np

def sigmoid(z):
    z = np.array(z)
    z = 1.0 / (1.0 + np.exp(-z))

    return z
