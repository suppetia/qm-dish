from abc import ABC, abstractmethod

import numpy as np


from dish.util.potential import fermi


class ChargeDistribution(ABC):

    def __init__(self, nucleus: "util.atom.Nucleus"):
        self.nucleus = nucleus

    @abstractmethod
    def __call__(self, r):
        ...

    @property
    @abstractmethod
    def R_rms(self):
        ...


class PotentialModel(ABC):

    def __init__(self, nucleus: "util.atom.Nucleus"):
        self.nucleus = nucleus

    @abstractmethod
    def __call__(self, r):
        ...


class CoulombPotential(PotentialModel):

    def __call__(self, r):
        return - self.nucleus.Z/r


class CoulombChargeDistribution(ChargeDistribution):

    @property
    def R_rms(self):
        return 0

    def __call__(self, r):
        return np.select(condlist=[r==0, r>0],
                         choicelist=[np.inf, 0])
                         # choicelist=[self.nucleus.Z, 0])


class FermiPotential(PotentialModel):

    def __call__(self, r):
        return - fermi.potential(Z=self.nucleus.Z,
                                 c=self.nucleus.R0,
                                 a=self.nucleus.a,
                                 r=r)


class FermiChargeDistribution(ChargeDistribution):

    @property
    def R_rms(self):
        # approximation given by Gustavson & M°artensson-Pendrill 1998
        return np.sqrt(3/5 * self.nucleus.R0**2 + 7/5 * np.pi**2 * self.nucleus.a**2)

    def __call__(self, r):
        return fermi.fermi_charge_distribution(self.nucleus.Z,
                                               self.nucleus.R0,
                                               self.nucleus.a,
                                               r)


class UniformBallPotential(PotentialModel):

    def __call__(self, r):
        # TODO: verify calculation
        # Zatsarinny and Froese Fischer, Computer Physics Communication 202 (2016), pp.287-303
        R = self.nucleus.R0 #* np.sqrt(5 / 3)
        return - np.select(condlist=[r < R, r >= R],
                           choicelist=[(self.nucleus.Z/R)*(3/2-r**2/(2*R**2)),
                                       self.nucleus.Z/r]
                           )


class UniformBallChargeDistribution(ChargeDistribution):

    @property
    def R_rms(self):
        return np.sqrt(3/5) * self.nucleus.R0

    def __call__(self, r):
        R = self.nucleus.R0 #* np.sqrt(5 / 3)
        rho0 = self.nucleus.Z/(4*np.pi*np.trapz(r[r<R]**2, x=r[r<R]))
        return np.select(condlist=[r < R, r >= R],
                         choicelist=[rho0, 0])

