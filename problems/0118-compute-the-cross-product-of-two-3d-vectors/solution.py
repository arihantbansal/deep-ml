import numpy as np

def cross_product(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return np.cross(a, b)