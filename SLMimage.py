import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os
import cv2
import scipy.ndimage
#from somewhere import dispSLMpattern, dispSLMimg, initializeSLMpx, findPxSLM, polygonInSquare
# =================================================================================================
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
# =================================================================================================
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
# =================================================================================================
def findPxSLM(px, a, n, pxSz, t, d, isDbg):
    t = t * np.pi
    d1 = d
    d2 = d + a

    xv, yv = polygonInSquare(t, a, d, n)
    v = np.arange(1, n + 1) - 0.5
    xq, yq = np.meshgrid(v, v)
    path = Path(np.column_stack((xv, yv)))
    in_ = path.contains_points(np.column_stack((xq.flatten(), yq.flatten()))).reshape(xq.shape)
    xv *= pxSz
    yv *= pxSz
    j, i = np.where(in_.T)
    pxIx = np.ravel_multi_index((j, i), (n, n))  # each value -1 of matlab but it's for indexing
    if isDbg:
        if 1:
            col = [0, 0, 0]
            col2 = [0, 0, 0]
            hp = plt.fill(xv, yv, col, edgecolor=col2, alpha=0.4)
        px[pxIx[:, 0], pxIx[:, 1]].set_facecolor([1, 0, 0])

    return px, pxIx
# =================================================================================================
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
    nps = math.ceil(((math.tan(tp) + 1) * n - d0 / math.cos(tp)) / (a / math.cos(tp)) / 2)  # # of stripes
    pxIx = []

    for i in range(1, nps+1):  # each stripe
        #memoryLimitPause()
        print(f"adding stripe: {i}/{nps}")

        d = d0 + 2 * a * (i - 1)
        px, pxix = findPxSLM(px, a, n, pxSz, t, d, isDbg)  # find px in stripe
        pxIx = np.append(pxIx, pxix, axis=0)
        
    return pxIx
# =================================================================================================
def dispSLMimg(n, pxIx, th, isDisp, isDbg):
    # n: # px
    # pxIx: positive px
    # rotation angle
    
    A = np.zeros(n**2)
    for index in pxIx:
        index = (np.rint(index)).astype(int)
        A[index] = 1
    A = np.reshape(A, (n, n), 'F')
    

    A = scipy.ndimage.rotate(A, th, reshape=True, order=0)
    # Okay so now we're gonna rotate because scipy.ndimage.rotate doesn't work, so this is gonna be manually done
    # A1 = np.array(A)
    #center = np.array(A.shape) // 2
    #angle = np.deg2rad(th)
    #rotation_matrix = np.array([[np.sin(angle), np.cos(angle)], [np.cos(angle), -np.sin(angle)]])   # Why this is it I do not know
    #coords = np.argwhere(A == 1)
    #translated_coords = coords - center
    #rotated_coords = np.dot(translated_coords, rotation_matrix)
    #restored_coords = (rotated_coords + center).astype(int)
    #rotated_size = np.max(np.abs(restored_coords), axis=0) * 2 + 1
    #rotated_size = np.ceil(rotated_size).astype(int)
    #rotated_matrix = np.zeros(rotated_size)
    #new_center = (rotated_size - 1) // 2
    #shifted_coords = restored_coords[:, ::-1] - center + new_center
    #rotated_matrix[shifted_coords[:, 0], shifted_coords[:, 1]] = 1
    #A = rotated_matrix
    # Still looks like shit but it feels a lot better
    
    # Too bad we rotate by -90, 0, 90 so scipy works fine
    
    if isDisp:
        plt.imshow(A, cmap='gray')
        plt.axis('off')
        plt.show()
    
    if isDbg:
        fig, ax = plt.subplots()
        ax.imshow(A, cmap='binary')
        ax.axis('off')
        plt.show()
    
    return A
# =================================================================================================

def SLMimage(a, ixDO, isPrint):
    # a: pattern half period
    # ixDO: 1-9
    # isPrint: comments printed
    # generates SLM images for simTIRF
    
    isDbg = 0
    isPrint = 1
    
    if 'a' not in locals():
        isDemo = 1
        isBlank = 0  # 1: if need to generate blank pattern
        
        if isBlank:
            # generates 0 intensity pattern (filtered by mask)
            a = 9
            ii = [1]
            jj = [1]
        else:
            a = 3
            ii = [1, 2, 3]
            jj = [1, 2, 3]
    else:
        isDemo = 0
        isBlank = 0
        ii = [(ixDO - 1) // 3 + 1]
        jj = [(ixDO - 1) % 3 + 1]
    
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
        folderOutPATH = 'C:/Users/hirka/Kural/Cell Reconstruction/SLM'
        folderBMPoutPATH = 'C:/Users/hirka/Kural/Cell Reconstruction/SLM/BMP/'
    else:
        folderOutPATH = 'D:/MATLAB/SIMdesignDATA/SLMpattern/'
        folderBMPoutPATH = 'D:/MATLAB/SIMdesignDATA/SLMpattern/BMP/'
        folderOutPATH = 'C:/Users/hirka/Kural/Cell Reconstruction/SLM'
        folderBMPoutPATH = 'C:/Users/hirka/Kural/Cell Reconstruction/SLM/BMP'
    # Add folder, BMP images
    folderBMPout = folderBMPoutPATH + cfgtxt
    imgBMPoutFN = folderBMPout + '/' + cfgtxt
    
    if not os.path.exists(folderBMPout):
        os.mkdir(folderBMPout)
    
    n = ny0
    pxSz = 1
    
    # Construct SLM --------------------------------------------
    for i in ii:  # theta
        t = T[i - 1]
        for j in jj:  # phi
            s = (j - 1) / 3  # phase shift
            k = (i - 1) * 3 + j
            fn = f'{imgBMPoutFN}{k}.bmp'
            if os.path.exists(fn):
                if isPrint:
                    print(f'already exists: {fn}')
            else:
                if isPrint:
                    print(f'generating: {fn}')
            pxIx = dispSLMpattern(a, n, pxSz, t, s, isDbg)
            A = dispSLMimg(n, pxIx, vaz[i - 1], 0, isDbg)  # FLIP
            if isBlank:
                A = np.transpose(A)
            A = A[:ny0, :nx0,]
            A = np.transpose(A, (1, 0)) # This is just a regular transpose i guess but written fancy?
            A = (A * 255).astype(np.uint8)  # Scale the values between 0 and 1 to 0-255 range
            A = cv2.cvtColor(A, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel grayscale image
            cv2.imwrite(fn, A)  # Save the image