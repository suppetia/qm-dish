import numpy as np
import scipy as sp
from math import factorial


def generalized_binomial_coefficient(alpha:float, k:int):
    r"""
    Return the generalized binomial coefficient
    \begin{pmatrix} \alpha \\ k \end{pmatrix} = \frac{1}{k!} \prod_{i=0}^{k-1} alpha - i
    :param alpha: arbitrary number
    :param k: positive integer
    """
    if k < 0:
        raise ValueError("k must be positive.")
    return np.prod(np.arange(k)-alpha)/factorial(k)


def count_nodes(data: np.array) -> int:
    """
    Count the number of nodes in data where a node is considered a zero crossing
    # (or a value close to zero if it is the first or last element).
    The algorithm to determine zero crossings is based on
        https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python .
    :param data: array to check the values for nodes
    :return: number of nodes
    """
    # bounds_are_nodes = np.isclose([data[0], data[-1]], 0)  # if value at boundaries are close to zero consider also as a node
    data_ = data[~np.isclose(data, 0, atol=1e-10)]  # remove values with exactly zero as a node is always a zero crossing if not at the boundaries
    pos = data_ > 0
    neg = ~pos
    return np.sum((pos[:-1] & neg[1:]) | (neg[:-1] & pos[1:]))  # + np.sum(bounds_are_nodes)

if __name__ == "__main__":
    import timeit

    # print(timeit.timeit(lambda: generalized_binomial_coefficient(12000, 100), number=10000))
    # print(timeit.timeit(lambda: sp.special.binom(12000, 100), number=10000))

    a = np.array([1e-9, 1,-1,2,0,-1,3,2,3, 0])
    print(count_nodes(a))
