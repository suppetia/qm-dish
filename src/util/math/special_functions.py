import numpy as np
import gmpy2

import timeit

from util.math import generalized_binomial_coefficient


def associated_legendre_polynomial(l: int, m: int, x: np.array):
    r"""
    Calculate the associated Legndre polynomial P_l^m(x) using the closed form
    P_l^m(x) = (-1)^m \cdot 2^l \cdot (1-x^2)^{m/2} \cdot \sum_{k=m}^l \frac{k!}{(k-m)!} \cdot x^{k-m}\cdot GBC[l,m] \cdot GBC[(l+k-1)/2, l]
    where GBC is the generalized binomial coefficient
    :param l: positive integer
    :param m: integer, -l <= m <= l
    :param x: positions to be evaluated
    """
    if l < 0:
        raise ValueError("l must be a positive integer.")
    if abs(m) > l:
        raise ValueError("m must be in range -l <= m <= l")
    sum_ = np.zeros_like(x, dtype=np.float64)
    for k in range(m, l+1):
        sum_ += np.math.factorial(k)/np.math.factorial(k-m) * np.power(x, k-m) \
                * generalized_binomial_coefficient(l, k) * generalized_binomial_coefficient((l+k-1)/2, l)
    return (-1)**m * 2**l * np.power(1-x**2, m/2) * sum_


def spherical_harmonic(l: int, m: int, theta, phi):
    r"""
    return the spherical harmonic Y_l^m(theta, phi) in defined as
    Y_l^m(\theta, \phi) = (-1)^m \sqrt{\frac{(2l+1)(l-m)!}{4\pi (l+m)!}} P_l^m(\cos\theta) e^{im\phi}
    where P_l^m is the associated Legendre polynom.
    :param l: positive integer
    :param m: integer, -l <= m <= l
    :param theta: polar angle
    :param phi: azimuthal angle
    """
    return (-1)**m * np.sqrt((2*l+1)*np.math.factorial(l-m)/(4*np.pi*np.math.factorial(l+m))) \
        * associated_legendre_polynomial(l, m, np.cos(theta)) * np.exp(1j * m * phi).real


def confluent_hypergeometric_f(a: int, b: int, x, order=150):
    r"""
    F(a,b,x) = \sum_{k=0}^{order} \prod_{i=0}^{k-1} \frac{1+i}{b+i} \frac{x^k}{k!}
    """
    if a <= 0:  # for non-positive values of 'a' F collapses into a polynomial of degree |a|
        order = min(order, -a)
    F_x = np.ones_like(x, dtype="float64")
    for k in range(1, order+1):
        F_x += (np.prod((np.arange(k, dtype=np.intc) + a) / (np.arange(k, dtype=np.intc) + b)) * np.power(x, k) / np.math.factorial(k)).astype(np.float64)
    return F_x

def confluent_hypergeometric_f2(a: int,b: int, x, order=10):
    r"""
    F(a,b,x) = \sum_{k=0}^{order} \prod_{i=0}^{k-1} \frac{1+i}{b+i} \frac{x^k}{k!}
    """
    summands = np.empty((order + 1, *x.shape))
    for k in range(order+1):
        summands[k] = (np.math.factorial(k+a-1)/np.math.factorial(k+b-1)) * (np.math.factorial(b-1) / np.math.factorial(a-1)) * (np.power(x, k)/np.math.factorial(k))
    return np.sum(summands, axis=0)


def confluent_hypergeometric_f4(a: int,b: int, x, order=10):
    r"""
    F(a,b,x) = \sum_{k=0}^{order} \prod_{i=0}^{k-1} \frac{1+i}{b+i} \frac{x^k}{k!}
    """
    summands = np.zeros_like(x, dtype="O")
    quotient_fac_b_a = gmpy2.fac(b) / gmpy2.fac(a)
    for k in range(order+1):
        summands += (gmpy2.fac(k+a)/gmpy2.fac(k+b)) * (np.power(x, k)/gmpy2.fac(k))
    return summands * quotient_fac_b_a

if __name__ == "__main__":
    a = -10
    b = 5
    x = np.linspace(0,100, num=10)

    # order = 10
    # print(confluent_hypergeometric_f(a,b,x, order=order))
    # print(confluent_hypergeometric_f2(a,b,x, order=order))
    # # print(confluent_hypergeometric_f4(a,b,x, order=order))
    #
    # print(timeit.timeit(lambda: confluent_hypergeometric_f(a,b,x, order=order), number=1000))
    # print(timeit.timeit(lambda: confluent_hypergeometric_f2(a,b,x, order=order), number=1000))
    # # print(timeit.timeit(lambda: confluent_hypergeometric_f4(a,b,x, order=order), number=1000))
    # # print(timeit.timeit(lambda: confluent_hypergeometric_f4b(a,b,x, order=order), number=1000))

    print(spherical_harmonic(0,0,x,0))

    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1)
    for l in range(5):
        plt.plot(x, associated_legendre_polynomial(l,0, x), label=f"l={l}, m=0")
    plt.legend()
    plt.show()
