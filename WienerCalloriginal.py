'''
So a couple of things need to be changed here:
	1) I cannot figure out the progress bar for the life of me, so I'm skipping it
	2) A lot of variables reference .mat files, which need to be changed to .py files
	3) The entirety of saveAllMAT is a mess
	4) The way I wrote WienerShiftParam() means I have to change all of the variables to be global
	   in both this script and also WienerShiftParam / WeighParam / Core
	5) I should probably use dictionaries instead of making f an object, but it's fine...
'''
import scipy.io
import numpy as np
import os
import glob
import shutil
import filecmp
import matplotlib.pyplot as plt
import warnings
import math
from PIL import Image
import cv2
from tqdm import tqdm
import gc
import pickle
import tkinter as tk
from tkinter import messagebox

# Functions I have to define
from visdiff import visdiff								# Done
from loadCFGrecon import loadCFGrecon
from isequalFile import isequalFile
from PSF2OTF import PSF2OTF
from fnGenericOTF import fnGenericOTF
from imreadstack import imreadstack
from reconStartFolders import reconStartFolders
from avger import avger
from genFN import genFN
from WienerShiftParam import WienerShiftParam
from WienerWeighParam import WienerWeighParam
from WienerCore import WienerCore
from displayReconParams import displayReconParams
from compareReconOTF import compareReconOTF
from tryexceptSkip import tryexceptSkip

class Object(object):
    pass

def WienerCall(cfgFN,f):
	# called by runEmul.m or runs standalone
	# __ input: _____________
	# cfgFN: cfg params (see '.\testData\cfgExp.m')
	# movie: [inFolder]\emulImg*.tif 
	# BG: [inFolder]\BG\BG*.tif
	# OTF: [inFolder]\OTF\OTF*.tif default:0

	F = [x.canvas.manager.canvas.figure
        for x in plt._pylab_helpers.Gcf.get_all_fig_managers()
        if x.canvas.manager.window.findChild(QtWidgets.QWidget, 'TMWWaitbar')]
	plt.close(F)

	# Algorithm Settings ===========================================================
	isFreqMask = 1	# default:1
	isSmoothPhaseDetection = 1	# phase angle fit and peak detection
	copyParamsDIR = []

	# Output Settings ==============================================================
	isHessian = 0
	testPSF = 0

	# Constants ====================================================================
	NpxOTF = 512
	NmPSF = 10 # px magnification

	# Experimental Input ===========================================================
	if 'cfgFN' in locals() and 'f' not in locals():
		cfgFNix = cfgFN
		dataDIR = 'D:\\DATA\\simTIRF\\Janelia__Kural_SIM__03_29_SUM_CALM_Electro\\'
		listData = glob.glob(dataDIR + '/**/acqImg*.tif', recursive=True)
		# Not sure if these lines mean much ...
		runEmul = 0
		cfgFNdir = 'C:\\Users\\hirak\\Kural\\Cell Reconstruction\\practice\\2\\'
		filename0 = 'acqImg.tif'
		cfgFN0 = 'cfgExp.m'	# Probably gonna change this to the python file
		f = Object()	# Pretty sure this works in creating f to be an object with attributes
		f.imgFolder = 'C:\\Users\\hirak\\Kural\\Cell Reconstruction\\practice\\2\\'
		f.imgBGfolder = 'D:\\MATLAB\\reconData\\dataBckgrnd\\bckgrnd_Huang2018\\'	# I dont think this is used
		f.imgOTFfolder = 'C:\\Users\\hirak\\Kural\\Cell Reconstruction\\reconData\\dataOTF\\'
		f.simFolder = f.imgFolder

		# Configure File
		cfgfn = 'cfgExp.m'	# Again, change to python file
		cfgFN = [f.imgFolder, 'cfgExp.m']
		cfgFN0 = [cfgFNdir, cfgFN0]

	elif 'cfgFN' not in locals():
		# Experimental Run =========================================================
		runEmul = 0
		cfgFNdir = 'C:\\Users\\hirak\\Kural\\Cell Reconstruction\\practice\\2\\'
		filename0 = 'acqImg.tif'
		f.imgFolder = 'C:\\Users\\hirak\\Kural\\Cell Reconstruction\\practice\\2\\'
		cfgFN0 = '\\cfgExp.m'
		f.imgBGfolder = 'D:\\MATLAB\\reconData\\dataBckgrnd\\bckgrnd_Huang2018\\'	# I dont think this is used
		f.imgOTFfolder = 'C:\\Users\\hirak\\Kural\\Cell Reconstruction\\reconData\\dataOTF\\'
		f.simFolder = f.imgFolder

		# Configure File
		cfgfn = 'cfgExp.m'
        cfgFN = [f.imgFolder, 'cfgExp.m']
        cfgFN0 = [cfgFNdir, cfgFN0]

    # Emulator Input ===============================================================
    elif not hasattr(f, 'isExpRun'):
    	runEmul = 1
    	cfgFN0 = [f.imgFolder, 'cfgEmulIMG.m']
        filename0 = 'emulImg2.tif'
        filename0 = 'emulImg2_noBckgrnd.tif'	# Seems redundant
        cfgFN = 'cfgSIM.m'
        cfgfn = cfgFN
        cfgFN = [f.simFolder, cfgFN];	# Keep lozg (configuration)
        if testPSF:
            f.imgOTFfolder = 'C:\\Users\\hirak\\Kural\\Cell Reconstruction\\reconData\\dataOTF\\'
    	else	# Outside call
        runEmul = 0
        cfgFN0 = cfgFN
        cfgfn = 'cfgOTHER.m'
        filename0 = f.filename0
        # Pretty sure this section doesn't do anything but I'll keep it anyway

    print(f'=== RUNNING: {f.imgFolder} ===')

    # Folder and Files =============================================================
    if hasattr(f, 'imgFolder2'):
    	pathnameIn = f.imgFolder2
    	outFolder = f.simFolder2
    else:
    	pathnameIn = f.imgFolder
    	outFolder = f.simFolder
    otfFolder = f.imgOTFfolder	# OTF
    bgFolder = f.imgBGfolder	# Background
    outFolder = outFolder		# Output
    if outFolder not in locals():
    	os.makedirs(outFolder)
    filenameList = [filename0]	# No cell arrays but a list seems like the same thing
    if not os.path.exists(pathnameIn + filename0):
    	mfn = Object()
    	mfn = sorted(glob.glob(pathnameIn + filename0[:-4] + '*.tif'))	# Not sure if sorted is needed
    	filenameSrch = {os.path.basename(f) for f in mfn}
    	filenameSrch = [os.path.splitext(f)[0] for f in filenameSrch]
    	filenameList = [f + '.tif' for f in filenameSrch]

    # Read Images ==================================================================
    fndbgDiffOrderLeak = 'dbgDiffOrderLeak.tif'
    if fndbgDiffOrderLeak in locals():
    	del fndbgDiffOrderLeak
    # I cannot tell you the purpose of these lines for the life of me

    # Read CFG =====================================================================
    if cfgFN not in locals():
    	print(f'--- cfgFN: (copied from) ---\n{cfgFN0}')
    	shutil.copyfile(cfgFN0, cfgFN)
    else:
    	if isequalFile(cfgFN0, cfgFN):	# uses isequalFile.py when made
    		print(f'--- cfgFN: (found) ---\n{cfgFN}')
    	else:
    		visdiff(cfgFN0,cfgFN)
    		plt.show()
    		inYN = input('--- Updated CFG file found ---\ncfgFN: Do you want to overwrite (overwrite uses local) [y/n]?\n')
    		if inYN == 'y':
    			shutil.copyfile(cfgFN0, cfgFN)
    			print(f'cfgFN: Local is overwritten')
    		else:
    			print(f'cfgFN: Local is used')

    [sys,isLoadReconParams,Aem,phaseShift,frmAvg,freqCutoff0,fcc0,fco0,freqGcenter,isBGcorrection,isOTF,PSFsigma,wienFilterCoeff,muHessian,sigmaHessian] = loadCFGrecon(cfgFN)
    runEstimation = 1
    copyReconParam = 0
    if isLoadReconParams:	# Skips param calculation, except nothing is written here ...
    elif copyParamsDIR:	# Runs if copyParamsDIR is not empty
		shftx_file = glob.glob(copyParamsDIR + '*_diffShftx.mat')[0]
		shfty_file = glob.glob(copyParamsDIR + '*_diffShfty.mat')[0]
		angleweigh_file = glob.glob(copyParamsDIR + '*_angleWeigh.mat')[0]
		shutil.copyfile(shftx_file, outFolder + '_diffShftxIN.mat')
		shutil.copyfile(shfty_file, outFolder + '_diffShftyIN.mat')
		shutil.copyfile(angleweigh_file, outFolder + '_angleWeighIN.mat')
		runEstimation = 0
		copyReconParam = 1
		warnings.warn(f'ReconParams will be loaded from: {copyParamsDIR}')
		# This assumes there is only a single file that ends in that way and also
		# I would need to change the end from .mat to a .py figure I think
	if isHessian == -1:
		runEstimation = 0
		copyReconParam = 0

	# Bunch of variables being defined
	bgflag = isBGcorrection	# Loads system background
	frmAvg0 = frmAvg	# For recon parameter detection
	rpRat = phaseShift	# SIM pattern phase shift [au, integers]
	wienCoeff = 2	# Wiener param1 (2: 3 rotations Hessian-SIM)
	if wienCoeff == 2:
		nphases = 3
		nangles = 3
	freqCutoff0 = freqCutoff0 * fcc0	# OTF mask radius [pxSIM]
	freqGcenter = freqGcenter
	freqGband = 50	# Half width of the band around 'freqGcenter' [px] (grating frequency)
	mu = muHessian	# Hessian param1
	sigma = sigmaHessian	# Hessian param2
	isReconInterleaved = 0
	loadSIMPattern = 0
	isSwapOPO = 0	# Default: 0
	frmAvg = loadSIMPattern + frmAvg0 * (loadSIMPattern == 0)	# loadSIMPattern --> frmAvg: 1
	twoPi = 2 * math.pi 	# You know, at first I thought this was dumb but I can see the utility
	frmMul = 1
	orderSwapVector = list(range(1, 10))	# No swap
	if isSwapOPO:
		orderSwapVector = [1, 4, 7, 2, 5, 8, 3, 6, 9]

	# Image information
	info = Image.open(pathnameIn + filenameList[0])
	starframe0 = 1
	zstack0 = len(info)
	sx, sy = info.size
	del info
	if frmAvg == 1:
		nEstimate = zstack0 // 9	# Number of estimates
	else:
		nEstimate = 1

	# BG and OTF ===================================================================
	if bgflag:
		fn = glob.glob(bgFolder + 'background*.tif')
		if len(fn) > 1:
			warnings.warn(f'multiple background*.tif, using: {fn[0].name}', UserWarning)
		bgname = os.path.basename(fn[0])

	if not testPSF and runEmul:
		PSF = cfgFN
		AvP = Aem / sys.Pxy
		[otf0,sigmaPSF] = PSF2OTF(PSF,NpxOTF,NmPSF)
		OTFfn0 = 'AvP{:.02f}px_sigmaPSF{:.02f}px'.format(AvP, sigmaPSF)	# We'll figure out if this works like I think it does
	else:	#Experimental
		[OTFfn0,sigmaPSF] = fnGenericOTF(otfFolder,sys,sx,isOTF)
		otf0 = imreadstack(OTFfn0)
		_, OTFfn0 = os.path.splitext(OTFfn0)	# Maybe ...
	otf0 = otf0.astype(np.uint16).astype(np.float64)
	nOTF = np.shape(otf0)[2]
	OTFfn0 = outFolder + 'OTF_' + OTFfn0 + '.tif'

	# Loops Call ===================================================================
	# Recon
	if isSmoothPhaseDetection:
		warnings.warn('isSmoothPhaseDetection:1 (default:0)')
	nMovie = len(filenameList)
	pathnameOutOTF = []
	if nOTF>1:
		pathnameOutOTF = f'{outFolder[:-1]}_OTF\\'
		os.makedirs(pathnameOutOTF)
	for j in range(5,7):	# You know, to this day I have no idea why 5 and 6 are chosen
		otf = otf0[:,:,j]
		PSFsgm = sigmaPSF[j]
		fcc = fcc0 / PSFsgm	# What is the point of this
		fco = fco0 / PSFsgm
		fcc = fcc0
		fco = fco0
	# Folder out
		if nOTF > 1:
			print(f'=== OTFno:{j}/{nOTF}===\n')
			pathnameOut0 = f'{pathnameOutOTF}{j:02d}_sigma{PSFsgm:.2f}\\'
			_, OTFfn0_, _ = os.path.splitext(OTFfn0)
			OTFfn = f'{pathnameOutOTF}{OTFfn0_}_ix{j:02d}.tif'
		else:
			OTFfn = OTFfn0
			pathnameOut0 = outFolder
		cv2.imwrite(OTFfn, otf.astype('uint16'))
		for i in range(1, nMovie + 1):
			print(f'=== running {i}/{nMovie} movie ===')
			if nMovie > 1:
				pathnameOut = f'{pathnameOut0[:-1]}_movie{i}\\'
			else:
				pathnameOut = pathnameOut0
			cfgFN2 = pathnameOut + cfgfn
			reconStartFolders(pathnameOut)
			if copyReconParam:
				shftx_file = glob.glob(copyParamsDIR + '*_diffShftx.mat')[0]
				shfty_file = glob.glob(copyParamsDIR + '*_diffShfty.mat')[0]
				angleweigh_file = glob.glob(copyParamsDIR + '*_angleWeigh.mat')[0]
				shutil.copyfile(shftx_file, pathnameOut + '_diffShftxIN.mat')
				shutil.copyfile(shfty_file, pathnameOut + '_diffShftyIN.mat')
				shutil.copyfile(angleweigh_file, pathnameOut + '_angleWeighIN.mat')
			if cfgFN != cfgFN2:
    			shutil.copyfile(cfgFN, cfgFN2)
    		saveAllMAT = f'{pathnameOut}saveAll.mat'

    		# Call =================================================================
    		filename = filenameList[i-1]
    		avger(filename,pathnameIn,pathnameOut)
    		if frmAvg == 1:
    			pathnameOutParams = genFN('dispParams',1,pathnameOut)
    			pathnameOutParams = pathnameOut + pathnameOutParams + '/'
    		else:
    			pathnameOutParams = pathnameOut
    		if not os.path.exists(pathnameOutParams):
    			os.makedirs(pathnameOutParams)
    		for ix in range(1, nEstimate + 1):
    			# Definitely need to change this to be python file data, not MATLAB
    			allvars = locals().items()
    			tosave = [var[0] for var in allvars if not var[1].__class__.__module__.startswith('matlab.ui') and not var[1].__class__.__module__.startswith('matlab.graphics')]
    			scipy.io.savemat(saveAllMAT, {var: locals()[var] for var in tosave})	# Saves it to a .mat file, but I don't want to deal with pickle, so...
    			Progressbar = tqdm(total=100, desc='Parameter Estimation...')	# I cannot promise that this progress bar works
    			starframe = starframe0 + (ix - 1) * 9
    			zstack = zstack0
    			if frmAvg == 1:
    				zstack = starframe + 9 - 1	# Why this is not simply 8 I do not know
    			try:
    				if runEstimation:
    					WienerShiftParam()	# I might have to define all of these variables as global variables
    				if runEstimation:
    					WienerWeighParam()
    				if isHessian >= 0:
    					WienerCore()
    				if isHessian:	# I don't think these functions exist, also isHessian = 0
    					# waitbar(0, Progressbar, 'Hessian reconstruction');
                        Bregman_Hessian_LowRam_2()
                        # waitbar(0, Progressbar, 'TV reconstruction');
                        Bregman_TV_denoise()
                        Running_average()
                    Progressbar.close()
                except Exception as e:
                	tryexceptSkip(e)
                try displayReconParams(filename,pathnameOutParams):
                except Exception as e:
                	tryexceptSkip(e)
                excluded = ['saveAllMAT']
                for var in [v for v in globals().keys() if v not in excluded]:
                	del globals()[var]
                gc.collect()
                with open(saveAllMAT, 'rb') as fileObject:
                	loaded = pickle.load(fileObject)
                os.remove(saveAllMAT)
                print(f'=== estimates: {ix}/{nEstimate} ===')
            cc = 3
        print(f'=== DONE: {nMovie} movies ===')
    if 0 and nOTF > 1:
    	fnrecon = 're-' + filename0[:-4] + '.tif'
    	ixOpt = compareReconOTF(pathnameOutOTF,fnrecon,sigmaPSF)
    	compareReconOTF_imageCrops(pathnameOutOTF,fnrecon,sigmaPSF)
    messagebox.showinfo('', 'All Done')