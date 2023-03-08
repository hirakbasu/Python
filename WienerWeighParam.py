def WienerWeighParam:
	excluded = ['inv_phase_matrix','orderSwapVector','illumFN','loadSIMPattern','isReconInterleaved','freqCutoff0','dataFN','pathnameOut','pathnameOutParams','pathnameIn','filename','psf','n','wienCoeff','wienFilterCoeff','rpRat','bgflag','bgname','starframe','frmAvg','frmMul','filename_all','wavelengh','twoPi','otf','zstack','sx','sy','sigma','mu','nphases','nangles','Progressbar','isFreqMask','isSmoothPhaseDetection','runEstimation','fcc','fco','saveAllMAT','isHessian']
	for var in [v for v in globals() if v not in excluded]:
		del globals()[var]