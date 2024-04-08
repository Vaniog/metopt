from common.utils import AbstractRunner, ExitCondition, Oracle, PlotConfig, Options
from common.utils import Vector
from common.utils import Metric
from numpy.linalg import inv
import numpy as np
import dataclasses
import typing as tp

from lab2.runners.newton_const import NewtonConstRunner, NewtonConstOptions


def search_up(f: tp.Callable[[float], float]):
    eps = 0.1
    f0 = f(0)
    while f(eps) < f0:
        f0 = f(eps)
        eps *= 2
    return eps


def trinary_search(f: tp.Callable[[float], float], x_max: float, iterations: int) -> float:
    lb = 0

    rb = x_max
    while iterations != 0:
        x1 = lb + (rb - lb) / 3
        x2 = x1 + (rb - lb) / 3

        if f(x1) > f(x2):
            lb = x1
        else:
            rb = x2

        iterations -= 1

    return lb


@dataclasses.dataclass
class NewtonSearchOptions(NewtonConstOptions):
    search_max: float = 10
    search_iterations: int = 10


class NewtonSearchRunner(NewtonConstRunner):
    opts: NewtonSearchOptions

    def _run(self, start: Vector, *args, **kwargs):
        prev = np.array(start.coords)
        cur = np.array(start.coords)
        while True:
            sk = self.sk(cur)

            def f(a: float):
                return self.o.f(*(cur + sk * a))

            cur = cur + self.sk(cur) * \
                  trinary_search(f, self.opts.search_max,
                                 self.opts.search_iterations)

            if self.opts.exit_condition(prev, cur):
                break
            prev = cur
