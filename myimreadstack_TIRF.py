# Reads a TIFF file specified by filename starting at frm1 until nImg images later
# Each frame is a 2D array slice in the returned 3D array img_read

import numpy as np
import tifffile as tf

def myimreadstack_TIRF(filename, frm1, nImg, testreadx, testready):
    with tf.TiffFile(filename) as t:  # Opens TIFF file to read and closes it after the loop
        img_read = np.zeros((testready, testreadx, nImg), dtype=np.uint16) # Unsigned 16-bit is to make sure values are correct
        for k in range(nImg):
            img_read[None, None, k] = t.pages[frm1+k].asarray() # Reads kth page and converts it to numpy array
    return img_read