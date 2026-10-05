cimport cython
import numpy as np
cimport numpy as np
import torch

@cython.boundscheck(False)
@cython.wraparound(False)
def fused_sdpa_inner(
    float[:, :, :, :] q,
    float[:, :, :, :] k,
    float[:, :, :, :] v,
    float scale,
    bint causal,
):
    cdef int B = q.shape[0]
    cdef int H = q.shape[1]
    cdef int S = q.shape[2]
    cdef int D = q.shape[3]
    cdef int i, j, b, h

    scores = np.zeros((B, H, S, S), dtype=np.float32)
    cdef float[:, :, :, :] sc = scores

    for b in range(B):
        for h in range(H):
            for i in range(S):
                for j in range(S):
                    if causal and j > i:
                        sc[b, h, i, j] = -1e9
                        continue
                    val = 0.0
                    for d in range(D):
                        val += q[b, h, i, d] * k[b, h, j, d]
                    sc[b, h, i, j] = val * scale

    return np.asarray(scores)
