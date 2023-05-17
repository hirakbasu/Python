import numpy as np
import math
import os
import cv2
from somewhere import dispSLMpattern, dispSLMimg

def SLMimage(a, ixDO, isPrint):
    isDbg = 0
    isPrint = 1
    
    if 'a' not in locals():
        isDemo = 1
        isBlank = 0  # 1: if need to generate blank pattern
        
        if isBlank:
            # generates 0 intensity pattern (filtered by mask)
            a = 9
            ii = 1
            jj = 1
        else:
            a = 3
            ii = [1, 2, 3]
            jj = [1, 2, 3]
    else:
        isDemo = 0
        isBlank = 0
        ii = (ixDO - 1) // 3 + 1
        jj = (ixDO - 1) % 3 + 1
    
    # Parameters -----------------------------------------------
    nx0 = 1536  # image size
    ny0 = 2048
    
    t0 = 1/4 + 1/18  # tilt angle for the first SLM angle [pi/rad]
    T = [t0 + 2/3 - 1/2, t0, t0 - 2/3 + 1/2]
    vaz = [-90, 0, 90]  # plot rotation angles for three SLM angles
    tSLM = np.add(np.add(np.multiply(T, 180), vaz), 90)
    tCAM = [math.fmod(180 - t, 180) for t in tSLM]
    tDiagonalDiff = t0 * 180 - 45 # I get excessive decimal precision like 10.000000000000007 but oh well
    tAxialDiff = 90 - T[0] * 180
    # cfgtxt = f'_{a}px({aCoeff})_{t0 * 180}deg'
    cfgtxt = f'{a:.2f}px_{tDiagonalDiff:.1f}deg'
    
    # Files ----------------------------------------------------
    if isDemo:
        folderOutPATH = 'D:/MATLAB/SIMdesignDATA/SLMpatternDEMO/'
        folderBMPoutPATH = 'D:/MATLAB/SIMdesignDATA/SLMpatternDEMO/BMP/'
    else:
        folderOutPATH = 'D:/MATLAB/SIMdesignDATA/SLMpattern/'
        folderBMPoutPATH = 'D:/MATLAB/SIMdesignDATA/SLMpattern/BMP/'
    # Add folder, BMP images
    folderBMPout = folderBMPoutPATH + cfgtxt
    imgBMPoutFN = folderBMPout + '/' + cfgtxt
    
    if not os.path.exists(folderBMPout):
        os.mkdir(folderBMPout)
    
    n = ny0
    pxSz = 1
    
    # Construct SLM --------------------------------------------
    for i in ii:  # theta, doesn't work for ii = integer, so if error make ii list of one number
        t = T[i - 1]
        for j in jj:  # phi, same as for jj = integer, make jj a list of one number
            s = (j - 1) / 3  # phase shift
            k = (i - 1) * 3 + j
            fn = f'{imgBMPoutFN}{k}.bmp'
            if os.path.exists(fn):
                if isPrint:
                    print(f'already exists: {fn}')
                continue
            else:
                if isPrint:
                    print(f'generating: {fn}')
            pxIx = dispSLMpattern(a, n, pxSz, t, s, isDbg)
            A = dispSLMimg(n, pxIx, vaz[i - 1], 0, isDbg)  # FLIP
            if isBlank:
                A = np.transpose(A)
            A = A[:ny0, :nx0, :]    # don't know if A is 3D, but if not it should be A = A[:ny0, :nx0]
            # A = np.transpose(A, (1, 0, 2)), we'll see which one is the right one
            A = np.transpose(A, (2, 1, 0))
            A = np.where(A != 0, 1, 0).astype(np.uint8) # idk, something about cv2 needing uint8 data type
            # write BMP
            cv2.imwrite(fn, A)