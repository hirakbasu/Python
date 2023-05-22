import numpy as np
import matplotlib.pyplot as plt
from somewhere import polygonInSquare

def findPxSLM(px, a, n, pxSz, t, d, isDbg):
    t = t * np.pi
    d1 = d
    d2 = d + a

    xv, yv = polygonInSquare(t, a, d, n)    # Fuck
    v = np.arange(1, n + 1) - 0.5
    xq, yq = np.meshgrid(v, v)
    path = Path(np.column_stack((xv, yv)))
    in_ = path.contains_points(np.column_stack((xq.flatten(), yq.flatten()))).reshape(xq.shape)
    xv *= pxSz
    yv *= pxSz
    i, j = np.where(in_)
    pxIx = np.column_stack((i, j))  #sub2ind so probably needs ravel_multi_index or something
    if isDbg:
        if 1:
            col = [0, 0, 0]
            col2 = [0, 0, 0]
            hp = plt.fill(xv, yv, col, edgecolor=col2, alpha=0.4)
        px[pxIx[:, 0], pxIx[:, 1]].set_facecolor([1, 0, 0])
    cc = 3

    return px, pxIx