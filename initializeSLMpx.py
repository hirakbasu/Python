import numpy as np
import matplotlib.pyplot as plt

def initializeSLMpx(n, pxSz):
    col = [1, 1, 1]
    col2 = [0, 0, 0]
    sRx = 0
    sRy = 0
    
    px = np.empty((n, n), dtype=object)
    
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            x1 = i - 1
            x2 = i
            y1 = j - 1
            y2 = j
            x = np.array([x1, x1, x2, x2]) + sRx
            y = np.array([y1, y2, y2, y1]) + sRy
            x *= pxSz
            y *= pxSz

            px[j - 1, i - 1] = plt.fill(x, y, col, edgecolor=col2)

    plt.axis('image')   # If needed to zoom into x and y like matlab, then plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel('nm')
    plt.ylabel('nm')

    return px