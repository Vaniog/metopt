from common.utils import AbstractRunner, Oracle
from common.utils import Vector
from numpy.linalg import inv
import numpy as np


class NewtonConstRunner(AbstractRunner):
    learning_rate: float

    def __init__(self, o: Oracle, start: Vector, learning_rate: float = 0.1):
        super().__init__(o, start)
        self.learning_rate = learning_rate

    def grad(self, p: np.ndarray, delta: float) -> np.ndarray:
        ds = []

        fp = self.o.f(*p)
        for i, _ in enumerate(p):
            coords = [*p]
            coords[i] += delta
            ds.append(self.o.f(*coords) - fp)

        res = np.array(list(map(lambda el: el / delta, ds)))
        return res

    def grad2(self, p: np.ndarray, delta: float) -> np.ndarray:
        dgs: np.ndarray = []

        gp = self.grad(p, delta)
        for i, _ in enumerate(p):
            coords = [*p]
            coords[i] += delta
            dgs.append(self.grad(np.array(coords), delta) - gp)

        res = np.array(list(map(lambda el: el * (1/delta), dgs)))
        return res

    def sk(self, p: np.ndarray, delta: float = 0.001) -> np.ndarray:
        grad2inv = inv(self.grad2(p, delta))
        a = grad2inv @ self.grad(p, delta)
        return -a

    def _run(self, start: Vector, *args, **kwargs):
        prev = np.array(start.coords)
        cur = np.array(start.coords)
        while True:
            cur = cur + self.sk(cur, self.learning_rate)
            if np.linalg.norm(prev - cur) < 0.001:
                break
            prev = cur


def f(x, y) -> float:
    return 2*(x-1)**2+y*y


ncr = NewtonConstRunner(Oracle(f, Vector(1, 0)), Vector(0, 0))
print(ncr.sk(np.array([3, 5])))

ncr.experiment()
print(1)
