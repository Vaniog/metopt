from common.utils import AbstractRunner, ExitCondition, Oracle
from common.utils import Vector
from common.utils import Metric
from numpy.linalg import inv
import numpy as np
import dataclasses

from lab2.runners.grad import grad, grad2, newton_dir


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
        return grad(self.o.f, p, self.opts.grad_delta)

    def grad2(self, p: np.ndarray) -> np.ndarray:
        return grad2(self.o.f, p, self.opts.grad_delta)

    def sk(self, p: np.ndarray) -> np.ndarray:
        return newton_dir(self.o.f, p, self.opts.grad_delta)

    def _run(self, start: Vector, *args, **kwargs):
        prev = np.array(start.coords)
        cur = np.array(start.coords)
        while True:
            cur = cur + self.sk(cur) * self.opts.learning_rate
            if np.linalg.norm(prev - cur) < self.opts.exit_condition_diff:
                break
            prev = cur
