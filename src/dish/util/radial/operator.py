import numbers
from numbers import Number
from typing import Callable, List, Union
from abc import ABC, abstractmethod

import numpy as np

from dish.util.math_util.linear_algebra import matmul_pointwise
from dish.util.radial.grid import DistanceGrid
from dish.util.radial.wave_function import RadialWaveFunction, RadialSchrodingerWaveFunction, RadialDiracWaveFunction
from dish.util.radial.integration import integrate_on_grid


class AbstractOperator(ABC):

    @abstractmethod
    def apply_on(self, ket: RadialWaveFunction) -> Union[RadialWaveFunction, np.float64, np.complex128]:
        ...

    def __mul__(self, other):
        if isinstance(other, RadialWaveFunction):
            return self.apply_on(other)
        elif isinstance(other, AbstractOperator):
            return _OperatorProduct(self, other)

        return NotImplemented

    def __add__(self, other):
        if isinstance(other, AbstractOperator):
            return _OperatorSum(self, other)

        return NotImplemented


class _OperatorChain(AbstractOperator, ABC):

    def __init__(self, op1: AbstractOperator, op2: AbstractOperator):

        self.operator_list = [op1, op2]

    def append(self, op: AbstractOperator):
        self.operator_list.append(op)

    @abstractmethod
    def apply_on(self, ket: RadialWaveFunction) -> RadialWaveFunction:
        ...


class MatrixOperator(AbstractOperator):

    def __init__(self,
                 operator_matrix: np.ndarray,
                 grid: DistanceGrid):
        self.op = operator_matrix
        self.grid = grid

    def apply_on(self, ket: RadialWaveFunction) -> RadialWaveFunction:
        assert self.grid == ket.grid
        assert self.op.shape[-1] == ket.Psi.shape[-1]

        if len(self.op.shape) == 1:
            # in 1-dimensional case (scalar wavefunction) use elementwise multiplication
            new_ket = self.op * ket.Psi
        else:
            new_ket = matmul_pointwise(self.op, ket.Psi)

        return type(ket)(ket.grid, new_ket, state=ket.state)


class ScalarOperator(MatrixOperator):
    def __init__(self,
                 operator_array: np.ndarray,
                 grid: DistanceGrid):
        if not len(operator_array.shape) == 1:
            raise ValueError(f"A ScalarOperator is represented as scalar values on a grid stored in an one-dimensional array, but the array is {len(operator_array.shape)}-dimensional.")
        assert len(operator_array) == grid.N
        super().__init__(operator_array, grid)


class BraOperator(AbstractOperator):

    def __init__(self, ket: RadialWaveFunction):
        self._ket = ket

    def apply_on(self, ket: RadialWaveFunction) -> Union[np.complex128, np.float64]:
        if not self._ket.grid == ket.grid:
            raise ValueError("Bra and Ket must be on the same grid")
        if isinstance(self._ket, RadialDiracWaveFunction):
            if not isinstance(self._ket, RadialDiracWaveFunction):
                raise ValueError("Ket must be a RadialDiracWaveFunction but is of type RadialSchrodingerWaveFunction")
            bra_conj = self._ket.Psi.astype(np.complex128)
            bra_conj[:, 0] *= -1j
            ket = ket.Psi.astype(np.complex128)
            ket[:, 0] *= 1j
            return integrate_on_grid(np.sum(bra_conj * ket, axis=1), grid=self._ket.grid)
        elif isinstance(self._ket, RadialSchrodingerWaveFunction):
            if not isinstance(self._ket, RadialSchrodingerWaveFunction):
                raise ValueError("Ket must be a RadialSchrodingerWaveFunction but is of type RadialDiracWaveFunction")
            bra_conj = self._ket.Psi
            return integrate_on_grid(bra_conj * ket.Psi, grid=self._ket.grid)

        return NotImplemented


class SymbolicScalarOperator(AbstractOperator):

    @abstractmethod
    def evaluate_on(self, ket: RadialWaveFunction, dim: int) -> np.ndarray:
        ...

    def apply_on(self, ket: RadialWaveFunction) -> RadialWaveFunction:
        if isinstance(ket, RadialSchrodingerWaveFunction):
            return RadialSchrodingerWaveFunction(r_grid=ket.grid, Psi=self.evaluate_on(ket, 1), state=None)
        elif isinstance(ket, RadialDiracWaveFunction):
            return DiagonalOperator([self]*ket.Psi.shape[1]).apply_on(ket)

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, RadialWaveFunction):
            return self.apply_on(other)
        elif isinstance(other, SymbolicScalarOperator):
            return _ScalarOperatorProduct(self, other)

        return NotImplemented

    def __add__(self, other):
        if isinstance(other, SymbolicScalarOperator):
            return _ScalarOperatorSum(self, other)

        return NotImplemented


class ProjectionOperator(SymbolicScalarOperator):

    def evaluate_on(self, ket: RadialWaveFunction, dim: int) -> np.ndarray:
        # return MatrixOperator(np.ones((ket.grid.N)), grid=ket.grid)
        dim_ket = 1 if len(ket.Psi.shape) == 1 else ket.Psi.shape[-1]
        if not 0 < dim <= dim_ket:
            raise ValueError(f"Dimension mismatch: trying to apply operator on dimension {dim} but the wavefunction has {dim_ket} dimensions")

        if dim_ket == 1:
            return ket.Psi
        if dim == 1:
            return (ket.Psi[:, 0]).astype(np.complex128)
        return ket.Psi[:, dim - 1]


class RadialOperator(SymbolicScalarOperator):

    def __init__(self, radial_func: Callable[[np.ndarray, ...], np.ndarray], /, fargs: set = (), nan_to_num=np.nan, inf_to_num=None):

        self.radial_func = radial_func
        self._fargs = fargs
        self._nan_to_num = nan_to_num
        self._inf_to_num = inf_to_num

    def evaluate_on(self, ket: RadialWaveFunction, dim: int) -> np.ndarray:

        values = np.nan_to_num(self.radial_func(ket.grid.r, *self._fargs),
                               nan=self._nan_to_num,
                               posinf=self._inf_to_num,
                               neginf=-self._inf_to_num if self._inf_to_num is not None else None)
        return values * ProjectionOperator().evaluate_on(ket, dim=dim)


class SymbolicMatrixOperator(AbstractOperator):

    def __init__(self, mat:Union[List[List[SymbolicScalarOperator]], np.ndarray]):

        self.dim = len(mat)
        # assert the matrix is quadratic
        for i in range(self.dim):
            assert self.dim == len(mat[i])
            # assert the types of the entries are valid
            for j in range(self.dim):
                if not isinstance(mat[i][j], (SymbolicScalarOperator, numbers.Number)):
                    raise ValueError(f"entries of SymbolicMatrixOperator must be either of type SymbolicScalarOperator or a number but one is of type {type(mat[i][j])}")

        self.mat = np.array(mat, dtype="object")

    def apply_on(self, ket: RadialWaveFunction) -> RadialWaveFunction:

        dim_ket = 1 if len(ket.Psi.shape) == 1 else ket.Psi.shape[-1]
        if not self.dim == dim_ket:
            raise ValueError(f"Mismatch of dimensions: the operator is {self.dim}x{self.dim} but the wavefunction is {dim_ket}")

        t = np.complex128 if dim_ket > 1 else np.float64
        new_ket = np.zeros((ket.grid.N, self.dim), dtype=t)

        for i in range(self.dim):
            for j in range(self.dim):
                scalar_op = self.mat[i,j]
                if isinstance(scalar_op, SymbolicScalarOperator):
                    new_ket[:, i] += scalar_op.evaluate_on(ket, dim=j + 1).astype(t)
                # elif isinstance(scalar_op, _OperatorChain):
                #     new_ket[:, i] += scalar_op.apply_on(ket)
                elif isinstance(scalar_op, Number):
                    if np.isclose(scalar_op, 0):
                        new_ket[:, i] += 0
                    else:
                        new_ket[:, i] += scalar_op * ProjectionOperator().evaluate_on(ket, dim=j+1).astype(t)
                else:
                    return NotImplemented

        if dim_ket == 1:
            new_ket = new_ket.flatten()

        return type(ket)(ket.grid, new_ket, state=None)


class DiagonalOperator(SymbolicMatrixOperator):

    def __init__(self, entries: Union[List[SymbolicScalarOperator], np.ndarray]):

        dim = len(entries)
        mat_repr = np.zeros((dim, dim), dtype="object")
        for i in range(dim):
            mat_repr[i,i] = entries[i]

        super().__init__(mat_repr)


class UnityOperator(DiagonalOperator):

    def __init__(self, dim: int = None):
        super().__init__([ProjectionOperator()]*dim)


class DifferentialOperator(SymbolicScalarOperator):

    def evaluate_on(self, ket: RadialWaveFunction, dim: int) -> np.ndarray:
        dim_ket = 1 if len(ket.Psi.shape) == 1 else ket.Psi.shape[-1]
        if not 0 < dim <= dim_ket:
            raise ValueError(f"Dimension mismatch: trying to apply operator on dimension {dim} but the wave function has {dim_ket} dimensions")

        temp_grid = DistanceGrid(ket.grid.h, ket.grid.r0, ket.grid.N + 1)

        # suppress numpy devide by zero warning
        old_settings = np.geterr()
        np.seterr(divide="ignore", invalid="ignore")
        if dim_ket == 1:
            r = ket.grid.r * np.nan_to_num(np.gradient(ket.Psi/ket.grid.r, temp_grid.r[1:] - ket.grid.r), nan=0)
        else:
            r = ket.grid.r * np.nan_to_num(np.gradient(ket.Psi[:, dim-1]/ket.grid.r, temp_grid.r[1:] - ket.grid.r), nan=0)
        np.seterr(**old_settings)
        return r


class _ScalarOperatorChain(_OperatorChain, SymbolicScalarOperator, ABC):

    def __init__(self, op1: SymbolicScalarOperator, op2: SymbolicScalarOperator):

        if not isinstance(op1, SymbolicScalarOperator) or not isinstance(op2, SymbolicScalarOperator):
            raise ValueError(f"op1 and op2 must be SymbolicScalarOperators")
        super().__init__(op1, op2)


class _OperatorProduct(_OperatorChain):

    def apply_on(self, ket: RadialWaveFunction) -> RadialWaveFunction:
        for op in reversed(self.operator_list):
            ket = op.apply_on(ket)

        return ket

    def __mul__(self, other: Union[AbstractOperator, RadialWaveFunction]):
        if isinstance(other, AbstractOperator):
            self.append(other)
            return self
        elif isinstance(other, RadialWaveFunction):
            return self.apply_on(other)

        return NotImplemented


class _ScalarOperatorProduct(_ScalarOperatorChain):

    def evaluate_on(self, ket: RadialWaveFunction, dim: int) -> np.ndarray:
        if isinstance(ket, RadialSchrodingerWaveFunction):
            res = RadialSchrodingerWaveFunction(ket.grid, ket.Psi.copy(), state=None, Psi_prime=ket.Psi_prime.copy())
            for op in reversed(self.operator_list):
                res.Psi[:] = op.evaluate_on(res, dim=dim)
            return res.Psi
        elif isinstance(ket, RadialDiracWaveFunction):
            res = RadialDiracWaveFunction(ket.grid, ket.Psi.copy().astype(np.complex128), state=None)
            for op in reversed(self.operator_list):
                res.Psi[:, dim-1] = op.evaluate_on(res, dim=dim)
            return res.Psi[:, dim-1]

        return NotImplemented

    def apply_on(self, ket: RadialWaveFunction) -> RadialWaveFunction:
        if isinstance(ket, RadialSchrodingerWaveFunction):
            for op in reversed(self.operator_list):
                ket = op.apply_on(ket)

            return ket
        return NotImplemented

    def __mul__(self, other: Union[AbstractOperator, RadialWaveFunction]):
        if isinstance(other, AbstractOperator):
            self.append(other)
            return self
        elif isinstance(other, RadialSchrodingerWaveFunction):
            return self.apply_on(other)

        return NotImplemented


class _OperatorSum(_OperatorChain):

    def apply_on(self, ket: RadialWaveFunction) -> RadialWaveFunction:
        dim_ket = 1 if len(ket.Psi.shape) == 1 else ket.Psi.shape[-1]
        t = np.complex128 if dim_ket > 1 else np.float64
        new_ket_values = np.zeros_like(ket.Psi, dtype=t)

        for op in self.operator_list:
            new_ket_values += op.apply_on(ket).Psi

        return type(ket)(r_grid=ket.grid, Psi=new_ket_values, state=None)

    def __add__(self, other):
        if isinstance(other, AbstractOperator):
            self.append(other)
            return self

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, RadialWaveFunction):
            return self.apply_on(other)

        return NotImplemented


class _ScalarOperatorSum(_ScalarOperatorChain):

    def evaluate_on(self, ket: RadialWaveFunction, dim: int) -> np.ndarray:
        dim_ket = 1 if len(ket.Psi.shape) == 1 else ket.Psi.shape[-1]
        t = np.complex128 if dim_ket > 1 else np.float64
        res = np.zeros(ket.Psi.shape[0], dtype=t)
        for op in self.operator_list:
            res += op.evaluate_on(ket, dim=dim)

        return res

    def apply_on(self, ket: RadialWaveFunction) -> RadialWaveFunction:
        if isinstance(ket, RadialSchrodingerWaveFunction):
            new_ket_values = np.zeros_like(ket.Psi)

            for op in self.operator_list:
                new_ket_values += op.apply_on(ket).Psi

            return type(ket)(r_grid=ket.grid, Psi=new_ket_values, state=None)
        return NotImplemented

    def __mul__(self, other: RadialWaveFunction):
        if isinstance(other, RadialWaveFunction):
            return self.apply_on(other)

        return NotImplemented
