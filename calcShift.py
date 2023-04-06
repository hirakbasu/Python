import numpy as np
# Calculates the magnitude of SIM vector shift

def calcShift(x, y):
    x = np.array(x).reshape(3, 3)   # Reshapes x and y from 9 column vector to 3x3 matrix
    y = np.array(y).reshape(3, 3)
    x2 = x - x[0]    # Sets first row to 0s to apparently make calculations easier
    y2 = y - y[0]
    
    r = np.sqrt(x2**2 + y2**2)  # Finds distance
    r = r[1:,:].reshape(-1,1)   # Removes first row and reshpes into column vector
    
    cc = 3  # No idea what cc means here, probably useless
    return r