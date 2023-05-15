import numpy as np
import tifffile

def imwritestack_appe(stack, filename):
    with tifffile.TiffWriter(filename, append=True) as t:
        tagstruct = {}
        tagstruct['ImageLength'] = stack.shape[0]
        tagstruct['ImageWidth'] = stack.shape[1]
        tagstruct['Photometric'] = 'minisblack'
        tagstruct['BitsPerSample'] = 32
        tagstruct['SampleFormat'] = 'float'
        tagstruct['PlanarConfiguration'] = 'contig'
        
        try:
            for k in range(stack.shape[2]):
                t.write(stack[:, :, k], metadata = tagstruct)
        except:
            t.write(stack[:, :], metadata = tagstruct)

    t.close()


'''
Example:
    
stack = np.random.rand(100,100,10)
filename = 'C:\\Users\\hirak\\Kural\\Cell Reconstruction\\Cell Images (TIFF)\\output.tif'
imwritestack_appe(stack, filename)

gonna be honest, i didn't wanna give up the functionality of it reading and appending
a whole stack at a time so i just made a try except block to cover for 2d images
'''