# I removed the dbgTxt because I don't think it really does anything
import numpy as np

def DOphase(rp, D, M, na, np):
    # na : nangles
    # np : nphases
    # M : inv_phase_matrix
    # D : dataFT
    # rp = 0

    if M.shape[2] == 1:
        M = np.tile(M, (1, 1, 3))

    # calculate filtered data
    for i in range(na):  # theta
        for j in range(np):  # phase
            jj = (i * np) + j
            for k in range(np):  # phase
                kk = (i * np) + k
                DMjk = M[j, k, i] * D[..., kk]
                rp[..., jj] += DMjk
    return rp