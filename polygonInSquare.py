import numpy as np

def polygonInSquare(t, a, d, n):
    d1 = d
    d2 = d + a
    x2 = []
    y2 = []
    x1 = d1 / np.cos(t)
    y1 = 0
    if x1 > n:
        y1 = (x1 - n) / np.tan(t)
        x1 = n
    x3 = d2 / np.cos(t)
    y3 = 0
    if x3 > n:
        y3 = (x3 - n) / np.tan(t)
        x3 = n
    if x1 != n and x3 == n:
        x2 = n
        y2 = 0
    x5 = []
    y5 = []
    y6 = d1 / np.sin(t)
    x6 = 0
    if y6 > n:
        x6 = (y6 - n) * np.tan(t)
        y6 = n
    y4 = d2 / np.sin(t)
    x4 = 0
    if y4 > n:
        x4 = (y4 - n) * np.tan(t)
        y4 = n
    if y6 != n and y4 == n:
        y5 = n
        x5 = 0
    if not np.size(x2):
        if not np.size(x5):  # x2 and x5 empty
            xv = np.array([x1, x3, x4, x6])
            yv = np.array([y1, y3, y4, y6])
        else:   # just x2 empty
            xv = np.array([x1, x3, x4, x5, x6])
            yv = np.array([y1, y3, y4, y5, y6])
    else:
        if not np.size(x5):  # just x5 empty
            xv = np.array([x1, x2, x3, x4, x6])
            yv = np.array([y1, y2, y3, y4, y6])
        else:   # none empty
            xv = np.array([x1, x2, x3, x4, x5, x6])
            yv = np.array([y1, y2, y3, y4, y5, y6])

    xv = np.clip(xv, 0, n)
    yv = np.clip(yv, 0, n)

    dbg = False
    if dbg:
        import matplotlib.pyplot as plt
        col = [1, 0, 0]
        col2 = [0, 0, 0]
        hp = plt.fill(xv, yv, facecolor=col, edgecolor=col2)
        plt.axis('image')
    return xv, yv