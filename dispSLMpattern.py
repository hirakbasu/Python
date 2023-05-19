import numpy as np
import math
from somewhere import initializeSLMpx, findPxSLM

def dispSLMpattern(a, n, pxSz, t, s, isDbg):
    # a: pattern width
    # n: # px
    # t: tilt angle pi/rad
    # s: shift

    d0 = s * a * 2  # distance from (0,0)
    if d0 > a:
        d0 = d0 - 2 * a

    # initialize
    px = []
    if isDbg:
        px = initializeSLMpx(n, pxSz)

    # switch pixels
    tp = t * math.pi
    np = math.ceil(((math.tan(tp) + 1) * n - d0 / math.cos(tp)) / (a / math.cos(tp)) / 2)  # # of stripes
    pxIx = []

    for i in range(1, np+1):  # each stripe
        memoryLimitPause()
        print(f"adding stripe: {i}//{np}")

        d = d0 + 2 * a * (i - 1)
        px, pxix = findPxSLM(px, a, n, pxSz, t, d, isDbg)  # find px in stripe
        pxIx.append(pxix)
        
    return pxIx