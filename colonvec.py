# Takes two lists n and M and returns a list of lists v
# where v[k] is a list of integers ranging from m[k] to M[k]

def colonvec(m, M):
    n = len(m)
    N = len(M)
    K = max(n, N)
    v = [None] * K
    if n == 1:
        m = [m[0]] * K
    elif N == 1:
        M = [M[0]] * K
    for k in range(K):
        v[k] = list(range(m[k], M[k] + 1))
    return v