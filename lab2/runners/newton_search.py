from common.utils import AbstractRunner, ExitCondition, Oracle, PlotConfig
from common.utils import Vector
from common.utils import Metric
from numpy.linalg import inv
import numpy as np
import dataclasses
import typing as tp

from lab2.runners.newton_const import NewtonConstRunner


@dataclasses.dataclass
class NewtonConstOptions:
    learning_rate: float = 1
    exit_condition_diff: float = 0.01
    grad_delta: float = 0.001


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
    search_iterations: float = 10


class NewtonSearchRunner(NewtonConstRunner):
    opts: NewtonSearchOptions

    def __init__(self, o: Oracle, start: Vector, opts: NewtonSearchOptions):
        super().__init__(o, start, opts)

    def _run(self, start: Vector, *args, **kwargs):
        prev = np.array(start.coords)
        cur = np.array(start.coords)
        while True:
            sk = self.sk(cur)

            def f(a: float):
                return self.o.f(*(cur + sk * a))
            cur = cur + self.sk(cur) *\
                trinary_search(f, self.opts.search_max,
                               self.opts.search_iterations)

            if np.linalg.norm(prev - cur) < self.opts.exit_condition_diff:
                break
            prev = cur
