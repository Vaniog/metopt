from common.utils import AbstractRunner, ExitCondition, Oracle
from common.utils import Vector
from common.utils import Metric
from numpy.linalg import inv
import numpy as np
import dataclasses


@dataclasses.dataclass
class NewtonConstOptions:
    learning_rate: float = 1
    exit_condition_diff: float = 0.01
    grad_delta: float = 0.001


class NewtonConstRunner(AbstractRunner):
    opts: NewtonConstOptions

    def __init__(self, o: Oracle, start: Vector, opts: NewtonConstOptions):
        super().__init__(o, start)
        self.opts = opts

    def grad(self, p: np.ndarray) -> np.ndarray:
        ds = []

        fp = self.o.f(*p)
        for i, _ in enumerate(p):
            coords = [*p]
            coords[i] += self.opts.grad_delta
            ds.append(self.o.f(*coords) - fp)

        res = np.array(list(map(lambda el: el / self.opts.grad_delta, ds)))
        return res

    def grad2(self, p: np.ndarray) -> np.ndarray:
        dgs: np.ndarray = []

        gp = self.grad(p)
        for i, _ in enumerate(p):
            coords = [*p]
            coords[i] += self.opts.grad_delta
            dgs.append(self.grad(np.array(coords)) - gp)

        res = np.array(
            list(map(lambda el: el * (1 / self.opts.grad_delta), dgs)))
        return res

    def sk(self, p: np.ndarray) -> np.ndarray:
        grad2inv = inv(self.grad2(p))
        a = grad2inv @ self.grad(p)
        return -a

    def _run(self, start: Vector, *args, **kwargs):
        prev = np.array(start.coords)
        cur = np.array(start.coords)
        while True:
            cur = cur + self.sk(cur) * self.opts.learning_rate
            if np.linalg.norm(prev - cur) < self.opts.exit_condition_diff:
                break
            prev = cur
