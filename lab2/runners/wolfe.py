import math
from common.utils import AbstractRunner, ExitCondition, Oracle, PlotConfig
from common.utils import Vector
from numpy.linalg import inv
import numpy as np
import dataclasses
import typing as tp
from lab2.runners.grad import grad
from lab2.runners.newton_const import NewtonConstOptions, NewtonConstRunner


@dataclasses.dataclass
class WolfeOptions(NewtonConstOptions):
    grad_delta: float = 0.001
    eps_armijo: float = 0.5
    eps_curvature: float = 0.75

    a0: float = 1
    teta_armija: float = 0.9


def wolfe_search(f, x: np.ndarray, s: np.ndarray, opts: WolfeOptions) -> float:
    fx = f(*x)
    dfsk = np.dot(grad(f, x, opts.grad_delta), s)

    def armijo(a: float) -> bool:
        return f(*(x + s * a)) - fx <= opts.eps_armijo * a * dfsk

    def curvature(a: float) -> bool:
        return np.dot(grad(f, x + s * a, opts.grad_delta), s) >= opts.eps_curvature * dfsk

    a = opts.a0
    teta_curvature = (1 + 1 / opts.teta_armija) / 2

    while True:
        if not armijo(a):
            a *= opts.teta_armija
            continue
        if not curvature(a):
            a *= teta_curvature
            continue
        break

    return a


class WolfeRunner(NewtonConstRunner):
    opts: WolfeOptions

    def _run(self, start: Vector, *args, **kwargs):
        prev = np.array(start.coords)
        cur = np.array(start.coords)
        while True:
            sk = self.sk(cur)

            cur = cur + self.sk(cur) * \
                  wolfe_search(self.o.f, cur, sk, self.opts)

            if self.opts.exit_condition(prev, cur):
                break
            prev = cur
