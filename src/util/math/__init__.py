import numpy as np
import scipy as sp


def generalized_binomial_coefficient(alpha:float, k:int):
    r"""
    Return the generalized binomial coefficient
    \begin{pmatrix} \alpha \\ k \end{pmatrix} = \frac{1}{k!} \prod_{i=0}^{k-1} alpha - i
    :param alpha: arbitrary number
    :param k: positive integer
    """
    if k < 0:
        raise ValueError("k must be positive.")
    return np.prod(np.arange(k)-alpha)/np.math.factorial(k)


if __name__ == "__main__":
    import timeit

    print(timeit.timeit(lambda: generalized_binomial_coefficient(12000, 100), number=10000))
    print(timeit.timeit(lambda: sc.special.binom(12000, 100), number=10000))
