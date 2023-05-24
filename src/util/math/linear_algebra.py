import numpy as np


def matmul_pointwise(A: np.array, B: np.array):
    """
    A and B should be arrays of matrices to multiply. Return the matrix product for each entry in the first dimension.
    :param A: shape (T, n, k)
    :param B: shape (T, k, m) or (T, m)
    :return: A[sym_t] @ B[sym_t] for 0 <= sym_t < T
    """
    assert A.shape[0] == B.shape[0]
    f = np.empty((A.shape[0], B.shape[-1]) if len(B.shape) == 2 else (A.shape[0], A.shape[1], B.shape[-1]))
    for i in range(A.shape[0]):
        f[i] = A[i] @ B[i]
    return f
