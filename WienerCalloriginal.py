# I assume I'll use these libraries and more later
import scipy
import numpy
import os
import shutil
import filecmp
class Object(object):
    pass

def WienerCall(cfgFN,f):
	# called by runEmul.m or runs standalone
	# __ input: _____________
	# cfgFN: cfg params (see '.\testData\cfgExp.m')
	# movie: [inFolder]\emulImg*.tif 
	# BG: [inFolder]\BG\BG*.tif
	# OTF: [inFolder]\OTF\OTF*.tif default:0

	F = findall(0, 'type', 'figure', 'tag', 'TMWWaitbar') # Dont know equivalent to have placeholder graphics variable
	del F

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
		listData = rdir([dataDIR, '\\**\\acqImg*.tif'])	# Don't know the equivalent to rdir
		# Not sure if lines 33 through 44 mean much ...
		runEmul = 0
		cfgFNdir = 'C:\\Users\\hirak\\Kural\\Cell Reconstruction\\practice\\2\\'
		filename0 = 'acqImg.tif'
		cfgFN0 = 'cfgExp.m'	# Probably gonna change this to the python file
		f = Object()	# Let it be known that I have no idea if this is how it works
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
    if outFolder not in locals():	# Pretty sure I could replace lines 98 - 100 with
    	os.makedirs(outFolder)		# if not os.path.isdir(outFolder): os.makedirs(outFolder)
    filenameList = [filename0]	# Python doesn't have cell arrays but this return a 0x0 empty cell array, so...
    if (pathnameIn + filename0) not in locals():
    	mfn = rdir((pathnameIn + filename0[1:len(filename0)-4]) '*.tif')	# Again, not sure about rdir
    	filenameSrch = [mfn.name]
    	[None, filenameSrch, None] = # cellfun(@fileparts,filenameSrch,'UniformOutput',false); I'll think about this later
    	filenameList = strcat(filenameSrch,'.tif')	# I know this is concatenating the two but mfn.name is blank...

    # Read Images ==================================================================
    fndbgDiffOrderLeak = 'dbgDiffOrderLeak.tif'
    if fndbgDiffOrderLeak in locals():
    	del fndbgDiffOrderLeak
    # This deletes the dbg file, if it exists

    # Read CFG =====================================================================
    if cfgFN not in locals():
    	print(f'--- cfgFN: (copied from) ---\n{cfgFN0}')
    	shutil.copyfile(cfgFN0, cfgFN)
    else:
    	if filecmp.cmp(cfgFN0, cfgFN, shallow=True):	# There is a isequalFile.m but I feel like this is better
    		print(f'--- cfgFN: (found) ---\n{cfgFN}')
    	else:
    		# visdiff(cfgFN0,cfgFN); drawnow      look into how visdiff returns the differences and how to update figures
    		inYN = input('--- Updated CFG file found ---\ncfgFN: Do you want to overwrite (overwrite uses local) [y/n]?\n')
    		if inYN == 'y':	# Something is off about this method
    			shutil.copyfile(cfgFN0, cfgFN)
    			print(f'cfgFN: Local is overwritten')
    		else:
    			print(f'cfgFN: Local is used')