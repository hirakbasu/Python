import math
import pickle
from calcShift import calcShift
import cv2
from scipy import ndimage
from scipy.signal import savgol_filter
from myimreadstack_TIRF import myimreadstack_TIRF
import warnings
from colonvec import colonvec
from DOphase import DOphase
from swap9frmOrder import swap9frmOrder
from regress_R2 import regress_R2

# Functions I have not defined yet
from emd import emd # This is super long and I don't want to do this right now

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
    
    # High Resolution OTF (512 --> 1024)
    K_h = np.array(H1_ang.shape)
    N_h = 2 * K_h
    L_h = np.ceil((N_h - K_h) / 2).astype(int)
    v_h = colonvec(L_h + 1, L_h + K_h)
    hw = np.zeros(N_h)
    hw[tuple(v_h)] = H_ang
    H_ang = hw
    hw[tuple(v_h)] = H_con
    H_con = hw
    hw[tuple(v_h)] = H1_ang
    H1_ang = hw
    hw[tuple(v_h)] = H1_con
    H1_con = hw
    
    # Diffraction Orders (DO) Phase Correction
    rp = DOphase(rp, dataFT, inv_phase_matrix, nangles, nphases)
    OTF_ang = H_ang
    OTF_con = H_con
    OTFmask_ang = H1_ang
    OTFmask_con = H1_con
    for rpi in range(1, (nangles)*(nphases-1)+1, 1):
        DO0 = rp[:,:,int(np.ceil(rpi/2))*3-2]
        DO1 = rp[:,:,int(np.ceil(rpi/2))+2*int(np.floor(rpi/2))-1]
        hw[tuple(v_h)] = DO1 # (512->1024)
        DO1 = hw
        hw[v_h] = DO0 # (512->1024)
        DO0 = hw
        kxtest = 2*np.pi*(diffShftx_512[rpi+1+int(np.floor((rpi-1)/2))-1]-n_512)/(2*n_512)
        kytest = 2*np.pi*(diffShfty_512[rpi+1+int(np.floor((rpi-1)/2))-1]-n_512)/(2*n_512)
        shftKernel[:,:] = np.exp(1j*(kxtest*xx2+kytest*yy2))
        # 1 OTF Mask Shift -- computes shifted OTFs for the angle and contrast channels, or something like that idk
        OTF_MASKshifted_ang[:,:] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(OTFmask_ang)))*shftKernel[:,:])))
        OTF_MASKshifted_con[:,:] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(OTFmask_con)))*shftKernel[:,:])))
        OTF_MASKshifted_ang[np.abs(OTF_MASKshifted_ang)>0.9] = 1
        OTF_MASKshifted_ang[np.abs(OTF_MASKshifted_ang)!=1] = 0 # Creates binary mask that masks out any high-frequency noise in the OTF
        OTF_MASKshifted_con[np.abs(OTF_MASKshifted_con)>0.9] = 1
        OTF_MASKshifted_con[np.abs(OTF_MASKshifted_con)!=1] = 0
        # 2 Shifted Masked OTF -- I think this shifts the OTF of the angle and contrast channel and applies the mask
        OTFshifted_ang = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(OTF_ang)))*shftKernel)))
        OTFshifted_con = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(OTF_con)))*shftKernel)))
        OTFshifted_ang *= OTF_MASKshifted_ang
        OTFshifted_con *= OTF_MASKshifted_con
        # 3 Diffraction Order 1 Shift
        DO1shifted = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(DO1)))*shftKernel)))
        # 4 Masked DO1 Shift Prop
        DO1overlap_ang = DO1shifted*OTF_MASKshifted_ang*OTF_ang
        DO1overlap_con = DO1shifted*OTF_MASKshifted_con*OTF_con
        # 5 Masked DO1 Shift Prop
        DO0overlap_ang = DO0*OTFshifted_ang
        DO0overlap_con = DO0*OTFshifted_con
        OTFshifted_OTFoverlap_ang[:,:,rpi-1] = OTFshifted_ang*OTF_ang  # used as a mask
        OTFshifted_OTFoverlap_con[:,:,rpi-1] = OTFshifted_con*OTF_con  # used as a mask
        DO1overlapNorm_ang[:,:,rpi-1] = DO1overlap_ang/(DO0overlap_ang+eps)  # the data to be masked
        DO1overlapNorm_con[:,:,rpi-1] = DO1overlap_con/(DO0overlap_con+eps)  # the data to be masked
        DO1overlap6_ang[:,:,rpi-1] = DO1overlap_ang  # only for regression
        DO0overlap6_ang[:,:,rpi-1] = DO0overlap_ang  # only for regression

    # If you couldn't tell, I kinda gave up trying to rationalize what exactly the code does...
    oDO1ang = np.angle(DO1overlapNorm_ang)
    oDO1abs = np.abs(DO1overlapNorm_con)
    xPhase = np.arange(-np.pi, np.pi + dPhase, dPhase)
    xAmp = np.arange(0, 0.6 + dAmp, dAmp)
    
    # Phase and Amplitude of Diffraction Patterns
    for ii in range(0, (nangles)*(nphases-1)):
        c = oDO1ang[:, :, ii]
        cc = oDO1abs[:, :, ii]
        d = np.ravel(c, order='F').reshape(-1)
        dd = np.ravel(cc, order='F').reshape(-1)
        
        a_ang = OTFshifted_OTFoverlap_ang[:, :, ii] # Larger mask for oDO1ang
        a_con = OTFshifted_OTFoverlap_con[:, :, ii] # Smaller mask for oDO1abs
        b_ang = np.ravel(a_ang, order='F').reshape(-1)
        b_con = np.ravel(a_con, order='F').reshape(-1)
        b_ang[b_ang!=0] = 1
        b_con[b_con!=0] = 1
        b_ang = np.column_stack((b_ang, d))
        b_con = np.column_stack((b_con, dd))
        b_ang = b_ang[b_ang[:, 0]!=0]
        b_con = b_con[b_con[:, 0]!=0]
        e = b_ang[:, 1]
        ee = b_con[:, 1]
        
        # Smoothing
        imf = emd(gHist)  # Empirical Mode Decomposition
        imf2 = emd(ggHist)
        isSmoothPhaseDetection = 1
        if isSmoothPhaseDetection:  # default: 0
            g = np.sum(imf[0:, :], axis=0)
            g = savgol_filter(g, window_length=11, polyorder=2) # This is the closest smoothing function python has to matlab's
            if np.count_nonzero(g == np.max(g)) > 1:
                pass    # This seems to be something that used to exist but doesn't anymore but I don't feel like deleting it
        elif np.max(gHist) < 50:
            g = np.sum(imf[4:, :], axis=0)
        else:
            g = np.sum(imf[3:, :], axis=0)
        gg = np.sum(imf2[0:, :], axis=0)
        
        # maximize(1) This maxes the window of a plot but we're not doing that
        p = g
        pp = gg
        h = np.where(g == np.max(g))[0]
        hh = np.where(gg == np.max(gg))[0]
        angle6[ii] = -np.pi + np.mean(h) * dPhase
        amp6[ii] = np.mean(hh) * dAmp
        dbgRegress[ii] = regress_R2(OTFshifted_OTFoverlap_ang[:,:,ii], DO1overlap6_ang[:,:,ii], DO0overlap6_ang[:,:,ii], Progressbar)
        with open(pathnameOut + filename[:-4] + '_angleWeigh.pickle', 'wb') as f:
            pickle.dump({'angle6': angle6, 'amp6': amp6}, f)
        
        # and then return some variabless, but I don't really know which ones are important