
"""
sigRMS: <type 'numpy.ndarray'>
Nb <type 'numpy.int64'>
c <type 'float'>
sRate <type 'numpy.uint16'>
fBaseline <type 'numpy.float64'>
"""
from __future__ import division

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def getBaseline_Cy(np.ndarray[np.float64_t,ndim=1] rmsVals, np.ndarray[np.float64_t,ndim=1] baseline, p_Nb, p_c, p_sRate, p_firstBaseline, p_len_rmsVals, p_lenBaseline):

	#assert rmsVals.dtype == DTYPE

	cdef int Nb = p_Nb
	cdef int nn, n
	cdef int sRate = p_sRate
	cdef double firstBaseline = p_firstBaseline
	cdef double c = p_c
	cdef int len_rmsVals = p_len_rmsVals
	cdef int lenBaseline = p_lenBaseline # Pre-compute number of chunks to emit
	cdef double old_bK = firstBaseline
	# cdef np.ndarray baseline = np.empty(lenBaseline, dtype=DTYPE)
	cdef double d_Nb = float(Nb)

	for nn,n in enumerate(range(Nb+1,len_rmsVals)):
		baseline[nn] = old_bK + (1/d_Nb)*(min(rmsVals[n-2*sRate],old_bK*c)-min(rmsVals[n-Nb-2*sRate],old_bK*c))
		old_bK = baseline[nn]

	return baseline