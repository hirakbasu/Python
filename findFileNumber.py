# returns the number and length of fn
# ex: reconParamWiener_002.tif returns (2,3)

def findFileNumber(fn):
    ix1 = fn.rfind('_')
    ix2 = fn.rfind('.')
    label = fn[ix1+1:ix2]
    nmLen = len(label)
    nm = int(label)
    return nm, nmLen