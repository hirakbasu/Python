import numpy as np
# Calculates the magnitude of SIM vector shift

def calcShift(x, y):
    x = x.reshape(3, 3) # Reshapes x and y into 3x3 matrix
    y = y.reshape(3, 3)
    x2 = x - x[0, :]    # Sets first row to 0s to apparently make calculations easier
    y2 = y - y[0, :]
    
    r = np.sqrt(x2**2 + y2**2)  # Finds distance, then transforms back into column vector
    r = r[1:, :].flatten()
    
    cc = 3  # No idea what cc means here, probably useless
    return r