import numpy as np

def swap9frmOrder(A, vs):
    # switches betw. phase amd orientation orders
    # POO: phase-orientation order (Default)
    # OPO: orientation-phase order

    isDemo = 0
    
    if A is None:
        vin = [1, 4, 7, 2, 5, 8, 3, 6, 9]
        A = np.ones((2, 2, 9)) * np.array(vin).reshape(1, 1, 9)
        vs = [1, 4, 7, 2, 5, 8, 3, 6, 9]  # swap vector
        Ain = A
        isDemo = 1
    
    if np.array_equal(vs, np.arange(1,10)):
        return A
    
    if A.shape[2] > 9:
        raise ValueError('9frm only')
    
    A = A[:, :, vs]
    
    if isDemo:
        Aout = A
        print(np.concatenate((Ain, np.nan * np.ones((2, 1, 9)), Aout), axis=1))
        A = None
        
    return A