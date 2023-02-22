import warnings

# Functions I have to define
from myimreadstack_TIRF import myimreadstack_TIRF
from swap9frmOrder import swap9frmOrder

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
		fd = myimreadstack_TIRF(dataFN,starframe,(nphases * nangles) * frmAvg,sx,sy)
		fd = swap9frmOrder(fd,orderSwapVector)

	# Check OTF and data size
