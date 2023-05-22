import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import math

def dispSLMimg(n, pxIx, th, isDisp, isDbg):
    # n: # px
    # pxIx: positive px
    # rotation angle
    
    A = np.zeros(n**2)
    for index in pxIx:
        A[index - 1] = 1
    square = int(math.sqrt(len(A)))
    A = [[A[j+i*square] for i in range(square)] for j in range(square)]

    # A = scipy.ndimage.rotate(A, th, reshape=True, order=0)
    # Okay so now we're gonna rotate because scipy.ndimage.rotate doesn't work, so this is gonna be manually done
    A1 = np.array(A)
    center = np.array(A1.shape) // 2
    angle = np.deg2rad(th)
    rotation_matrix = np.array([[np.sin(angle), np.cos(angle)], [np.cos(angle), -np.sin(angle)]])   # Why this is it I do not know
    coords = np.argwhere(line == 1)
    translated_coords = coords - center
    rotated_coords = np.dot(translated_coords, rotation_matrix)
    restored_coords = (rotated_coords + center).astype(int)
    rotated_size = np.max(np.abs(restored_coords), axis=0) * 2 + 1
    rotated_size = np.ceil(rotated_size).astype(int)
    rotated_matrix = np.zeros(rotated_size)
    new_center = (rotated_size - 1) // 2
    shifted_coords = restored_coords[:, ::-1] - center + new_center
    rotated_matrix[shifted_coords[:, 0], shifted_coords[:, 1]] = 1
    A = rotated_matrix
    # Still looks like shit but it feels a lot better
    
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