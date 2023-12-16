import numpy as np
# import yaml

from dataclasses import dataclass

from util.atomic_units import a_0
from util.potential import FermiPotential, CoulombPotential, UniformBallPotential
from util.potential import FermiChargeDistribution, CoulombChargeDistribution, UniformBallChargeDistribution

@dataclass
class Nucleus:
    Z: int
    M: float
    R_rms: float  # root-mean-square radius
    a: float = 2.3e-15/a_0 / (4*np.log(3))  # default value from Parpia and Mohanty, Phys.Rev.A, 46 (1992), Number 7

    @property
    def mu(self):
        return 1/(1+1/self.M)

    @classmethod
    def construct_from_name(cls, name):
        # load from yaml file
        ...

    def potential(self, r, model="Fermi"):
        if model.lower() in ["f", "fermi"]:
            return FermiPotential(self)(r)
        elif model.lower() in ["u", "uniform", "ball", "uniformball"]:
            return UniformBallPotential(self)(r)
        elif model.lower() in ["point", "point-like", "pointlike", "p", "coulomb", "c"]:
            return CoulombPotential(self)(r)
        else:
            raise ValueError(
                f"'model' must be either 'Fermi', 'uniform' or 'point-like' but is {model}.")

    def CoulombPotential(self, r):
        return self.potential(r, model="Coulomb")
    def FermiPotential(self, r):
        return self.potential(r, model="Fermi")
    def UniformBallPotential(self, r):
        return self.potential(r, model="uniform")


    def charge_distribution(self, r, model):
        if model.lower() in ["f", "fermi"]:
            return FermiChargeDistribution(self)(r)
        elif model.lower() in ["u", "uniform", "ball", "uniformball"]:
            return UniformBallChargeDistribution(self)(r)
        elif model.lower() in ["point", "point-like", "pointlike", "p", "coulomb", "c"]:
            return CoulombChargeDistribution(self)(r)
        else:
            raise ValueError(
                f"'model' must be either 'Fermi', 'uniform' or 'point-like' but is {model}.")

    def FermiChargeDistribution(self, r):
        return self.charge_distribution(r, model="Fermi")
    def CoulombChargeDistribution(self, r):
        return self.charge_distribution(r, model="Coulomb")
    def UniformBallChargeDistribution(self, r):
        return self.charge_distribution(r, model="uniform")