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

        for k in range(stack.shape[2]):
            t.write(stack[:, :, k], metadata=tagstruct)

    t.close()


'''
Example:
    
stack = np.random.rand(100,100,10)
filename = 'output.tif'
imwritestack_appe(stack, filename)

So the whole tagstruct thing is to make sure the metadata of each slice is preserved, and apparently just
appending each slice can lead to some metadata loss. But tbh I can't see the difference and I'm not
convinced I set up the tagstruct here correctly but it seems alright.
'''