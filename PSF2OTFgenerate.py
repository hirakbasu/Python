import numpy as np
from scipy.fftpack import fftshift, fft2
from skimage.io import imsave

def PSF2OTFgenerate():
    # Output filenames
    OTFfn = 'C:/Users/hirak/Kural/Cell Reconstruction/recon/imgfolder/reconData/dataOTF/OTF.tif'
    PSFfn = 'C:/Users/hirak/Kural/Cell Reconstruction/recon/imgfolder/reconData/dataOTF/PSF.tif'

    # PSF parameters
    gausKernelSz = 17
    gausKernelSg = np.linspace(1.80, 1.99, 20)  # Generate list of values, spacing should be (end - start)/0.03 + 1
    gausKernelSg = np.linspace(1, 6, 21)
    gausKernelSg = np.linspace(1.5, 2, 11)
    gausKernelSg = np.linspace(1.2, 2.2, 41)
    #gausKernelSg = np.linspace(0.5, 1.175, 28)
    #gausKernelSg = np.linspace(0.1, 2.2, 43)    # 0.05 spacing
    # If you change this, change ffr.fnGenericOTF sigmaPSF

    PSF_stack = []
    OTF_stack = []

    for i in range(len(gausKernelSg)):
        # Generate PSF
        PSF = generate_gaussian(gausKernelSz + 1, gausKernelSg[i])
        PSF_stack.append(PSF)

        # Convert PSF to OTF
        OTF = fftshift(fft2(PSF, (2048, 2048)))
        OTF = np.abs(OTF)
        OTF = np.uint16(OTF / np.max(OTF) * (2 ** 16 - 1))
        OTF_stack.append(OTF)

    # Save PSF stack
    PSF_stack = np.array(PSF_stack)
    imsave(PSFfn, PSF_stack, plugin='tifffile')

    # Save OTF stack
    OTF_stack = np.array(OTF_stack)
    imsave(OTFfn, OTF_stack, plugin='tifffile')

def generate_gaussian(size, sigma):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    center = size // 2
    gauss = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    return gauss / np.sum(gauss)  # Normalize the Gaussian

PSF2OTFgenerate()
