import cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def helper(double[:] pv_orig, int m1):
    cdef int i,j
    cdef double[:] pvm = np.zeros(m1)
    cdef double[:] alpha = np.zeros(m1+1)
    cdef double[:] alpha2 = np.zeros(m1)
    cdef double[:] ghat2 = np.zeros(m1)
    cdef double[:] S = np.zeros(m1+1)
    cdef double[:] ghat = np.zeros(m1+1)
    
    alpha[0] = 0
    for i in range(0, m1):
        pvm[i] = max(pv_orig[0:(i+1)])
        
    # calculate model size
    with cython.nogil:
        for i in range(1, m1+1):
            alpha[i] = pvm[i-1]
            alpha2[i-1] = pvm[i-1] - 0.0000001
    return (np.array(pvm), np.array(alpha), np.array(alpha2))