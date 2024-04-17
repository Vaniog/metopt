import abc

import numpy as np
from OptimizationTestFunctions.functions import easy_bounds
from numpy import cos, exp, pi

from common.utils import Vector


class Base(abc.ABC):
    x_best: np.ndarray
    dim: int

    @property
    def __name__(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def call(self, *args):
        ...

    def target(self) -> Vector:
        return Vector(*self.x_best)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            raise Exception("???")
        if len(args) == 1:
            return self.call(*args[0])
        else:
            return self.call(*args)


class Basic(Base):
    def __init__(self, dim: int):
        self.x_best = np.zeros(dim)
        self.bounds = easy_bounds(5)
        self.dim = dim

    def call(self, *args):
        return sum(map(lambda a: a ** 2, args))


class Diagonal(Base):
    def __init__(self, dim):
        self.x_best = np.zeros(2)
        self.bounds = easy_bounds(5)
        self.dim = 2

    def call(self, x, y):
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


class FlatWithMin(Base):
    def __init__(self, dim):
        self.x_best = np.array([pi, pi])
        self.bounds = easy_bounds(5)
        self.dim = 2

    def call(self, x, y):
        return -cos(x) * cos(y) * exp(-((x - pi) ** 2 + (y - pi) ** 2))
