import numpy as np


def matmul_pointwise(A: np.ndarray, B: np.ndarray):
    """
    A and B should be arrays of matrices to multiply. Return the matrix product for each entry in the first dimension.
    :param A: shape (T, n, k)
    :param B: shape (T, k, m) or (T, k)
    :return: A[sym_t] @ B[sym_t] for 0 <= sym_t < T
    """
    assert A.shape[0] == B.shape[0]
    assert A.shape[-1] == B.shape[1]
    t = np.float64 if A.dtype == np.float64 and B.dtype == np.float64 else np.complex128
    f = np.empty((A.shape[0], A.shape[1]) if len(B.shape) == 2 else (A.shape[0], A.shape[1], B.shape[-1]), dtype=t)
    for i in range(A.shape[0]):
        f[i] = A[i] @ B[i]
    return f

if __name__ == "__main__":
    N = 1000000
    M = 2
    import time
    A = np.zeros((N,M,M), dtype=np.complex128)
    A[:,0,0] = np.arange(N)
    A[:,0,1] = np.arange(N)*1j
    A[:,1,0] = -np.arange(N)*1j
    A[:,1,1] = 1
    for i in range(M):
        A[:, i, 0] = (i*1j)**i

    b = np.vstack([np.arange(N)]*M).T
    print(b.shape)

    t_start = time.perf_counter()
    res = matmul_pointwise(A, b)
    print(time.perf_counter()-t_start)

    t_start = time.perf_counter()
    res2 = np.zeros((A.shape[0], A.shape[1]), dtype=A.dtype)
    for i in range(A.shape[1]):
        for j in range(A.shape[2]):
            res2[:, i] += A[:, i, j] * b[:, j]

    print(time.perf_counter() - t_start)
    print(np.allclose(res, res2))

