'''
A couple of things here:
	1) Variables need to be defined as global
	2) It is enirely possible that some lists inside for-loops are indexed at 1 instead of 0
	3) I am deeply dissappointed by the absence of love of the twoPi variable
'''

import warnings
from skimage.transform import resize
import numpy as np
import math
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

# Functions I have to define
from myimreadstack_TIRF import myimreadstack_TIRF
from swap9frmOrder import swap9frmOrder
from DOphase import DOphase
from dft import dft
from colonvec import colonvec
from dispFreqbandPos import dispFreqbandPos
from diffOrderOverlap import diffOrderOverlap

def WienerShiftParam():
	excluded = ['freqLimCalc','orderSwapVector','loadSIMPattern','isReconInterleaved','freqCutoff0','freqGcenter','freqGband','pathnameOut','pathnameOutParams','pathnameIn','filename','otf','rpRat','wienCoeff','wienFilterCoeff','bgflag','bgname','starframe','frmAvg','frmMul','filename_all','wavelengh','twoPi','zstack','sx','sy','sigma','mu','nphases','nangles','Progressbar','isFreqMask','isSmoothPhaseDetection','runEstimation','fco','fcc','saveAllMAT','isHessian']
	for var in [v for v in globals() if v not in excluded]:
		del globals()[var]

	# Read Data
	dbg = 0
	dataFN = pathnameIn + filename
	illumFN = pathnameIn + filename[:-4] + "_patternSIM.tif"
	patternFreqFN = pathnameOutParams + filename[:-4] + '_patternFreq.tif'
	imageFourierFN1 = pathnameOutParams + filename[:-4] + '_imageFourier1.tif'
	imageFourierFN2 = pathnameOutParams + filename[:-4] + '_imageFourier2.tif'
	if loadSIMPattern:
		fd = myimreadstack_TIRF(illumFN,1,9,sx,sy)
		fd = swap9frmOrder(fd,orderSwapVector)
	else:
		if (starframe + (nphases * nangles) * frmAvg - 1 > zstack):
			frmAvg2 = (zstack - starframe + 1) / (nphases * nangles)
			warnings.warn(f'# of averages will be set to {frmAvg2} (smaller than requested:{frmAvg})', Warning)
			frmAvg = frmAvg2
		fd = myimreadstack_TIRF(dataFN,starframe,(nphases * nangles) * frmAvg, sx, sy)
		fd = swap9frmOrder(fd,orderSwapVector)

	# Check OTF and Data Size
	otf = resize(otf, (512, 512), order = 1, mode = 'reflect', anti_aliasing = True)
	H = otf
	fdd = np.zeros((sx, sy, nphases * nangles))
	n = max(sx, sy, 512)
	if 0:
		# code that never sees the light of day
	fc = fcc * math.ceil(220 * (n / 512))
	freqLim = math.ceil(freqGcenter * (n / 512))
	freqBand = math.ceil(freqGband * (n / 512))
	if n > 512:
		H = resize(H, (n, n), order = 1, mode = 'reflect', anti_aliasing = True)
	phase_matrix0 = [[0, 0, 0], [rpRat[0] / sum(rpRat), 1, -rpRat[0] / sum(rpRat)], [(rpRat[0] + rpRat[1]) / sum(rpRat), 1, -(rpRat[0] + rpRat[1]) / sum(rpRat)]]
	phase_matrix = math.exp(1j * twoPi * phase_matrix0)	# May have to define or transform phase_matrix0 to numpy array or list to use math module
	for frm in range(0, nphases * nangles):
		fdd[:, :, frm] = np.sum(fd[:, :, frm:(nphases * nangles):(nphases*nangles)*frmAvg], axis=2) / frmAvg
	fd = fdd

	# Initialize
	test = np.zeros((4, 1))
	diffShftx = np.zeros((nphases*nangles, 1))
	diffShftx[:, 0] = n
	diffShfty = np.zeros((nphases*nangles, 1))
	diffShfty[:, 0] = n
	DOcorrNorm6 = np.zeros((2*n-1, 2*n-1, (nangles)*(nphases-1)))
	k_x, k_y = np.meshgrid(np.arange(-(n)//2, (n+1)//2), np.arange(-(n)//2, (n+1)//2))	# Creates n x n grid of spacial frequencies
	k_r = np.sqrt(k_x**2 + k_y**2)
	indi = k_r > fc
	H[indi] = 0
	H = np.abs(H)
	inv_phase_matrix = np.linalg.inv(phase_matrix)
	rp = np.zeros((n, n, nphases*nangles))
	rpt = np.zeros((2*n,2*n,nphases*nangles))

	# Data Fourier Transform
	fd512 = np.zeros((n, n))
	K_h = [fd.shape[0], fd.shape[1]]
	N_h = (n, n)
	L_h = np.ceil((np.subtract(N_h, K_h))/2).astype(int)
	v_h = colonvec(L_h+1, L_h+K_h)
	hw = np.zeros((N_h))
	for ii in range(0, nphases*nangles):
		hw[tuple(v_h)] = fd[:, :, ii]	# No idea if this works but let's do it
		fd512[:, :, ii] = hw
	dataFT = fftshift(fft2(ifftshift(fd512)))
	H1 = H
	H1[H1 != 0] = 1	# Creates mask where non-zero elements replaced by 1
	H2 = H1
	H9 = np.tile(H1, (1, 1, nphases*nangles))	# Copies the H1 filter accross the 9 stacks
	if 0:
		# more code that doesn't run
	dataFT = H9 * dataFT

	# H1
	K_h = H1.shape
	N_h = 2 * K_h
	L_h = np.ceil((np.subtract(N_h, K_h))/2).astype(int)
	v_h = colonvec(L_h+1, L_h+K_h)
	hw = np.zeros((N_h))
	hw[v_h] = H
	H = hw
	hw[v_h] = H1
	h1 = hw
	rp = DOphase(rp,dataFT,inv_phase_matrix,nangles,nphases)
	rp = rp / (np.abs(rp) + np.finfo(float).eps)
	if 0:
		# bunch of junk that doesn't run

	# Giant Loop Time
	if plt.fignum_exists(6344):
		plt.close(6344)
	for rpi in range(1, (nangles)*(nphases-1) + 1, 2):
		errCoeffx = 1
		errCoeffy = 1
		rp = np.multiply(rp, H9)
		ix1[rpi - 1] = math.ceil(rpi/2)*nphases - 1	# [2, 5, 8]
		ix2[rpi - 1] = math.ceil(rpi/2) + 2*math.floor(rpi/2)	# [1, 4, 7]
		DO0 = rp[:, :, ix1[rpi - 1]]
		DO1 = rp[:, :, ix2[rpi - 1]]

		# Initial Estimate maxx and maxy
		H2flip = H2[::-1, ::-1]
		H2corr = dft(H2, H2flip)
		H2corrBW = H2corr
		H2corrBW[H2corrBW < 0.9] = 0
		H2corrBW[H2corrBW != 0] = 1

		# Search Illum Pattern Freq Band Position
		DO1_ = DO1.conj()
		DO1_ = DO1_[::-1, ::-1]
		DOcorr=dft(DO0,DO1_)
		DOcorrMasked = np.multiply(DOcorr, H2corrBW)
		DOcorrNorm0 = np.abs(DOcorrMasked / (H2corr + np.finfo(float).eps))
		k_x, k_y = np.meshgrid(np.arange(-(2*n)//2+1, (2*n)//2), np.arange(-(2*n)//2+1, (2*n)//2))
		k_r = np.sqrt(k_x**2 + k_y**2)
		clip = ((k_r < (freqLim - freqBand)) | (k_r > (freqLim + freqBand)))
		DOcorrNorm = DOcorrNorm0
		DOcorrNorm[clip] = 0
		yy, xx = np.unravel_index(np.argmax(DOcorrNorm), DOcorrNorm.shape)
		maxx = xx
		maxy = yy
		dispFreqbandPos(xx,yy,rpi, DOcorrNorm0,DOcorrNorm,freqLim,freqBand, patternFreqFN)	# Keep in mind rpi = 1, so if used for indexing, rpi = rpi - 1

		# Spatial Freq Vectors
		ky = twoPi * (maxy - n) / (2 * n)	# I worked so hard to make the variable twoPi so you bet your ass I'm using it
		ky = twoPi * (maxx - n) / (2 * n)	# No peasant 2 * math.pi will be used here
		x = list(range(2*n))
		y = np.arange(2*n).reshape((2*n, 1))	# I could write y = np.transpose(list(range(2*n))) but allegedly this is more efficient
		xx2 = np.tile(x, (2*n,1))
		yy2 = np.tile(y, (1,2*n))

		hw[tuple(v_h)] = DO1
		DO1 = hw
		hw[tuple(v_h)] = DO0
		DO0 = hw

		# Test 1
		shftKernel[:, :] = np.exp(1j*(kx*xx2 + ky*yy2))
		oo = diffOrderOverlap(shftKernel, H1, H, DO1, DO0, maxx, maxy, dbg)
		maxx_tmp1 = maxx - 1e-5
		maxx_tmp2 = maxx + 1e-5
		maxy_tmp1 = maxy - 1e-5
		maxy_tmp2 = maxy + 1e-5
		kx_tmp1 = twoPi * (maxx_tmp1 - n) / (2 * n)	# I hate how you go through the effort of defining twoPi and proceed to never use it
		kx_tmp2 = twoPi * (maxx_tmp2 - n) / (2 * n)
		ky_tmp1 = twoPi * (maxy_tmp1 - n) / (2 * n)
		ky_tmp2 = twoPi * (maxy_tmp2 - n) / (2 * n)
		for ii in range(0, 4):	# Are you kidding me, python doesn't have its own switch statement it's so slow
			if ii == 0:
				kxtest = kx_tmp1
				kytest = ky
			elif ii == 1:
				kxtest = kx_tmp2
				kytest = ky
			elif ii == 2:
				kxtest = kx
				kytest = ky_tmp1
			else:
				kxtest = kx
				kytest = ky_tmp2
			shftKernel = np.exp(1j * (kxtest * xx2 + kytest * yy2))
			test[ii] = diffOrderOverlap(shftKernel, H1, H, DO1, DO0, maxx, maxy, dbg)
		if test[0] > test[1]:
			flag_maxx = -1
		elif test[0] < test[1]:
			flag_maxx = 1
		else:
			Progressbar.close()
			warnings.warn('Can not estimate the pattern wave vector')
			return
		if test[2] > test[3]:
			flag_maxy = -1
		elif test[2] < test[3]:
			flag_maxy = 1
		else:
			Progressbar.close()
			warnings.warn('Can not estimate the pattern wave vector')
			return
		maxx_tmp = maxx
		maxy_tmp = maxy

		# While Loop
		wix = 0
		while (errCoeffx > 1e-4) or (errCoeffy > 1e-4):
			# Maxy
			maxy_tmp1 = maxy - 1e-5
			maxy_tmp2 = maxy + 1e-5
			ky_tmp1 = twoPi * (maxy_tmp1 - n) / (2 * n)
			ky_tmp2 = twoPi * (maxy_tmp2 - n) / (2 * n)
			for ii in range(2, 4):
				if ii == 2:
					kxtest = twoPi * (maxx - n) / (2 * n)
					kytest = ky_tmp1
				else:
					kxtest = twoPi * (maxx - n) / (2 * n)
					kytest = ky_tmp2
				shftKernel = np.exp(1j * (kxtest * xx2 + kytest * yy2))
				test[ii] = diffOrderOverlap(shftKernel, H1, H, DO1, DO0, maxx, maxy, dbg)
			if test[2] > test[3]:
				flag_maxy = -1
			elif test[2] < test[3]:
				flag_maxy = 1
			else:
				flag_maxy = -1 * flag_maxy
			while (errCoeffx > 1e-4):
				maxy_tmp = maxy + flag_maxy * errCoeffx
				kytest = twoPi * (maxy_tmp - n) / (2 * n)
				kxtest = twoPi * (maxx - n) / (2 * n)
				shftKernel = np.exp(1j * (kxtest * xx2 + kytest * yy2))
				oo_tmp = diffOrderOverlap(shftKernel, H1, H, DO1, DO0, maxx, maxy, dbg)
				if oo_tmp <= oo:
					errCoeffx = 0.5 * errCoeffx
				else:
					oo = oo_tmp
					maxy = maxy_tmp
					break
			MAXY[wix, (rpi + 1) // 2] = maxy

			# Maxx
			maxx_tmp1 = maxx - 1e-5
			maxx_tmp2 = maxx + 1e-5
			kx_tmp1 = twoPi * (maxx_tmp1 - n) / (2 * n)
			kx_tmp2 = twoPi * (maxx_tmp2 - n) / (2 * n)
			for ii in range(0, 2):
				if ii == 0:
					kxtest = kx_tmp1
					kytest = twoPi * (maxy - n) / (2 * n)
				else:
					kxtest = kx_tmp2
					kytest = twoPi * (maxy - n) / (2 * n)
				shftKernel = np.exp(1j * (kxtest * xx2 + kytest * yy2))
				test[ii] = diffOrderOverlap(shftKernel, H1, H, DO1, DO0, maxx, maxy, dbg)
			if test[0] > test[1]:
				flag_maxx = -1
			elif test[0] < test[1]:
				flag_maxx = 1
			else:
				flag_maxx = -1 * flag_maxx
			while (errCoeffy > 1e-4):
				maxx_tmp = maxx + flag_maxx * errCoeffy
				kytest = twoPi * (maxy - n) / (2 * n)
				kxtest = twoPi * (maxx_tmp - n) / (2 * n)
				shftKernel = np.exp(1j * (kxtest * xx2 + kytest * yy2))
				oo_tmp = diffOrderOverlap(shftKernel, H1, H, DO1, DO0, maxx, maxy, dbg)
				if oo_tmp <= oo:
					errCoeffy = 0.5 * errCoeffy
				else:
					oo = oo_tmp
					maxx = maxx_tmp
					break
			MAXX[wix, (rpi + 1) // 2] = maxx
			wix += 1
# ___________________________________________________________________________________________________________
		diffShftx[rpi + math.floor((rpi-1)/2), :] = maxx
		diffShfty[rpi + math.floor((rpi-1)/2), :] = maxy
		diffShftx[rpi + 1 + math.floor((rpi-1)/2), :] = 2 * n - maxx
		diffShfty[rpi + 1 + math.floor((rpi-1)/2), :] = 2 * n - maxy
		DOcorrNorm6[:, :, rpi - 1] = DOcorrNorm
		Progressbar.update(rpi / ((nangles) * (nphases - 1) * 100)
		Progressbar.set_description('Grating Vector Parameter Estimate...')

	# Save (get exact syntax)
	np.savetxt(pathnameOut + filename[:-4] + '_diffShftx.txt', diffShftx)	# .txt could work but maybe try .py
	np.savetxt(pathnameOut + filename[:-4] + '_diffShfty.txt', diffShfty)
	Progressbar.update(100)
	Progressbar.set_description('Vector Parameters Set')