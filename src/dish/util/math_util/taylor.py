import sympy as sp
from math import factorial

sym_t, sym_x = sp.symbols("t x")


def _evaluate_func_at(func, var, x0):
    # in case there is a singularity in at x0 calculate the limit instead of just replacing var with x0
    primitive_solution = func.subs({var: x0})
    if primitive_solution == sp.nan:
        return sp.limit(func, var, x0)
    else:
        return primitive_solution


def taylor_coefficients(func, order, x0, var=sym_x):
    """
    return the coefficients a_n = f^{(n)} (x0) / n! of the Taylor series expansion in x0
    :param func: symbolic function to expand in <variable>
    :param order: order of the taylor series expansion
    :param var: variable to expand <func> in
    :return:
    """

    terms = [_evaluate_func_at(func, var, x0)]

    d_func = func.subs({var: sym_t})

    for i in range(1, order+1):
        d_func = d_func.diff(sym_t)
        # print(d_func)
        term = _evaluate_func_at(d_func, sym_t, x0) / factorial(i)
        terms.append(term)

    return terms


if __name__ == "__main__":

    j = sp.symbols("j")

    func = sym_x/sp.log(1 - sym_x)
    func2 = -sp.log(1 - sym_x) * (1 - sym_x) ** j

    taylor_func = taylor_coefficients(func, order=3, x0=0)
    print(taylor_func)
    for t in taylor_func:
        print(sp.fraction(t))
    print(taylor_coefficients(func2, order=5, x0=0))


