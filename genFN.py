import os
import glob
from findFileNumber import findFileNumber

# fn: filename or foldername
# FN: parent foldername
# ex: genFN('dispParam', 1, 'C:\\Users\\hirak\\Kural\\Cell Reconstruction\\practice\\2\\') ==> dispParam_01

def genFN(fn, isFolder = False, FN = './'):
    PWD = os.getcwd()   # returns current working directory
    os.chdir(FN)        # changes current working directory to FN (default is the same)
    if not FN.endswith(os.sep):
        FN += os.sep    # checks if FN ends with \ and if it doesn't, well now it does
    if isFolder:
        fn = fn + '_01' # adds _01 to fn
        fnSrch = glob.glob(FN + fn + '*')   # searches all files in FN that start with fn and returns a list of file paths
        if len(fnSrch) == 0:
            ix = 0
        else:
            fn2 = fnSrch[-1].name   # gets last element's name
            ix = int(fn2[len(fn):]) # takes string starting from the end of fn2 and makes that ix
        while os.path.exists(FN + fn):
            ss = fn.split('_')
            ix = int(ss[-1]) + 1
            ixTxt = f'{ix:02}'      # formats ix as a 2 digit number with a leading 0
            fn = fn[:-2] + ixTxt
    else:
        # searches files starting with fn except last 4 characters that end with the last 4 characters of fn in FN
        # os.path.join makes sure  the directory path is formatted correctly (/ or \ depending on the system)
        # glob.glob returns list of file paths that match, recursive is to also search subdirectories
        fnSrch = glob.glob(os.path.join(FN, f"{fn[:-4]}*{fn[-4:]}"), recursive=True)
        if len(fnSrch) == 0:
            ix = 0
        else:
            ix, _ = findFileNumber(fnSrch[-1].name) # returns the number and length, ex: reconParamWiener_002.tif returns (2,3)
        ixTxt = f'_{ix+1:02d}'  # string that is the new file number formatted to two digits with leading zeros
        fn = fn[:len(fn)-4] + ixTxt + fn[len(fn)-4:]    # replaces fn _ix with _ix+1
        os.chdir(PWD)   # changes current wording directory to PWD
    return fn