import math
import pickle
from calcShift import calcShift
import cv2
from scipy import ndimage
from myimreadstack_TIRF import myimreadstack_TIRF
import warnings
from colonvec import colonvec

# Functions I have not defined yet
from swap9frmOrder import swap9frmOrder

def WienerWeighParam:
	excluded = ['inv_phase_matrix','orderSwapVector','illumFN','loadSIMPattern','isReconInterleaved','freqCutoff0','dataFN','pathnameOut','pathnameOutParams','pathnameIn','filename','psf','n','wienCoeff','wienFilterCoeff','rpRat','bgflag','bgname','starframe','frmAvg','frmMul','filename_all','wavelengh','twoPi','otf','zstack','sx','sy','sigma','mu','nphases','nangles','Progressbar','isFreqMask','isSmoothPhaseDetection','runEstimation','fcc','fco','saveAllMAT','isHessian']
	for var in [v for v in globals() if v not in excluded]:
		del globals()[var]
    
    # OTF Frequency Cuttoff and Sampling
    n_512 = max([512, sx, sy])
    fcANG = 120
    if wavelength == 488:
        fcABS = 105 # Default
    elif wavelength == 561:
        fcABS = 120
    elif wavelength == 647:
        fcABS = 80
    fc_con = math.ceil(fco * fcABS * (n_512 / 512))
    fc_ang = math.ceil(fco * fcANG * (n_512/512))
    dPhase = 0.02   # Sampling distance (phase)
    dAmp = 0.02     # Sampling distance (amplitude)
    nphases = 3
    
    # Load diffShftMAT, assuming they follow the format '_diffShftx.pickle'
    with open(pathnameOut + filename[:-4] + '_diffShftx.pickle', 'rb') as f:
        diffShftx = pickle.load(f)
    with open(pathnameOut + filename[:-4] + '_diffShfty.pickle', 'rb') as f:
        diffShfty = pickle.load(f)
    # Stores new x and y shifts scaled based on n_512 by taking average of shftx and shfty, basically accounts for change in image size
    diffShftx_512 = diffShftx * (n_512 // floor((diffShftx[0,0]+diffShfty[0,0])/2))
    diffShfty_512 = diffShfty * (n_512 // floor((diffShftx[0,0]+diffShfty[0,0])/2))
    shiftMag = calcShift(diffShftx_512, diffShfty_512)
    sMinMax = np.array([np.min(shiftMag), np.max(shiftMag)])    # Returns 1x2 matrix array([min, max]) of shiftMag
    overlapSize = shiftMag - 2*np.array([fcANG, fcABS])
    
    # H1  H9 (OTF) -- don't ask me what H1 and H9 means, it's probably some mathematical notation
    H_ang = otf
    H_ang = cv2.resize(H_ang, (n_512, n_512), interpolation=cv2.INTER_LINEAR)   # Bilinear interpolation
    H_con = H_ang
    
    k_x, k_y = np.meshgrid(np.arange(-(n_512)//2, (n_512)//2), np.arange(-(n_512)//2, (n_512)//2))  # Creates coordinate grid n_512 x n_512 centered at origin
    k_r = np.sqrt(k_x**2 + k_y**2)  # Computes magnitude of the frequency values at each point in the meshgrid
    indi_ang = k_r > fc_ang     # Creates boolean mask for values in Fourier transform with frequencies greater than fc_angle
    indi_con = k_r > fc_con     # Same deal, but for values in the Fourier transform with frequencies greater than fc_con
    H_ang[indi_ang] = 0 # All values greater than fc_ang are 0
    H_con[indi_con] = 0 # Value greater than fc_con are 0
    H_ang = np.abs(H_ang)   # Takes absolute value of Fourier transform
    H_con = np.abs(H_con)   # which is important since transform can have negative values
    H1_ang = H_ang
    H1_con = H_con
    H1_ang[H1_ang != 0] = 1 # All non-zero values are 1
    H1_con[H1_con != 0] = 1 # which creates a binary mask for the OTF
    
    if loadSIMPattern:  # Loads illumination pattern specified by 'illumFN'
        fd = myimreadstack_TIRF(illumFN, 1, 9, sx, sy)  # Reads 9-frame TIRF, returns result as array
        fd = swap9frmOrder(fd, orderSwapVector) # Swaps frame order based on orderSwapVector (which I think is the normal 1:9)
    else:
        if starframe + (nphases * nangles) * frmAvg - 1 > zstack:
            Progressbar.close() # I feel like we've given up on the progress bar but oh well its here, I've grown an attachment
            warnings.warn('The number of averaged frames should be smaller', Warning)
            return
        else:
            fd = myimreadstack_TIRF(dataFN, starframe, (nphases * nangles) * frmAvg, sx, sy)
            fd = swap9frmOrder(fd, orderSwapVector)
    for frm in range(0, nphases*nangles, 1):
        # This averages together chunks of frmAvg on consecutive frames to reduce noise, I think
        fdd_an[:, :, frm] = np.sum(fd[:, :, frm:(nphases*nangles):(nphases*nangles)*frmAvg], axis=2) / frmAvg
    fd = fdd_an
    
    fd512 = np.zeros((n_512, n_512))
    K_h = [fd.shape[0], fd.shape[1]]
    N_h = [n_512, n_512]
    L_h = np.ceil((np.array(N_h) - np.array(K_h)) / 2).astype(int)
    v_h = colonvec(L_h+1, L_h+K_h)
    hw = np.zeros(N_h)
    
    if bgflag == 1:
        bg = imreadstack(bgname)    # Loads bgname image stack
        bckgrnd = bg[(end//2)+1-(sx)//2:(end//2)+(sx)//2, (end//2)+1-(sy)//2:(end//2)+(sy)//2]
    elif bgflag == 0:
        bckgrnd = np.zeros((sx, sy))
    for ii in range(nphases*nangles):
        fd[:, :, ii] = fd[:, :, ii] - bckgrnd   # This subtracts background
        # Copies 2D slice of fd into hw at index ii at corresponding indices specified by v_h and pads the edges with 0s
        # ex: fd = [1,2,3], [4,5,6], [7,8,9] has its central 2x2 block copied to a 4x4 hw, so K_h = [3,3] and N_h = [4,4]
        # the result is [0,0,0,0], [0,5,6,0], [0,8,9,0], [0,0,0,0] except now repeated along the third dimension
        hw[v_h[0]:v_h[1]+1, v_h[2]:v_h[3]+1] = fd[:, :, ii]
        fd512[:, :, ii] = hw
    
    # Fourier Transform
    dataFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fd512)))  # Yeah you're right this is crazy how this isn't used
    
    # Initialize
    amp6 = np.zeros((3 * (nphases - 1), 1))
    angle6 = np.zeros((3 * (nphases - 1), 1))
    OTFshifted_OTFoverlap_con = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)))
    OTFshifted_OTFoverlap_ang = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)))
    DO1overlapNorm_con = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)))
    DO1overlapNorm_ang = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)))
    DO1overlap6_ang = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)))
    DO0overlap6_ang = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)))
    rp = np.zeros((n_512, n_512, nphases * nangles))
    temp_separated = np.zeros((n_512, n_512, nphases))
    x = np.arange(2 * n_512)
    y = np.arange(2 * n_512)[:, np.newaxis]
    xx2 = np.tile(x, (2 * n_512, 1))
    yy2 = np.tile(y, (1, 2 * n_512))