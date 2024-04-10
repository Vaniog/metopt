from common.utils import AbstractRunner, ExitCondition, Oracle, Options
from common.utils import Vector
from common.utils import Metric
from numpy.linalg import inv
import numpy as np
import dataclasses

from lab2.runners.grad import grad, grad2, newton_dir


@dataclasses.dataclass
class NewtonConstOptions(Options):
    learning_rate: float = dataclasses.field(default=1, metadata={"bounds": (0, 2)})
    grad_delta: float = dataclasses.field(default=0.001, metadata={"bounds": (0, 0.1)})

    def validate(self):
        return all((
            0 < self.learning_rate,
            0 < self.grad_delta,
            super().validate(),
        ))


class NewtonConstRunner(AbstractRunner):
    opts: NewtonConstOptions

    def grad(self, p: np.ndarray) -> np.ndarray:
        return grad(self.o.f, p, self.opts.grad_delta)

    def grad2(self, p: np.ndarray) -> np.ndarray:
        return grad2(self.o.f, p, self.opts.grad_delta)

    def sk(self, p: np.ndarray) -> np.ndarray:
        return newton_dir(self.o.f, p, self.opts.grad_delta)

    def _run(self, start: Vector, *args, **kwargs):
        prev = np.array(start.coords)
        cur = np.array(start.coords)
        while self.running():
            cur = cur + self.sk(cur) * self.opts.learning_rate
            if self.opts.exit_condition(prev, cur):
                break
            prev = cur
