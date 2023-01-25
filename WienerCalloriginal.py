import scipy
import numpy
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
        filename0 = 'emulImg2_noBckgrnd.tif'
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

    print(f'=== RUNNING: {f.imgFolder} ===')	# Dont know why the f is typed fancy but oh well

    # Folder and Files =============================================================
    if hasattr(f, 'imgFolder2'):
    	pathnameIn = f.imgFolder2
    	outFolder = f.simFolder2
    else:
    	pathnameIn = f.imgFolder
    	outFolder = f.simFolder
    otfFolder = f.imgOTFfolder	# OTF
    bgFolder = f.imgBGfolder	# Background