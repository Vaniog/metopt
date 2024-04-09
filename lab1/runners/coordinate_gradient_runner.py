from common.utils import *
import typing as tp
from .gradient_descent_runner import GradientDescendRunner


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


class CoordinateGradientRunner(OldRunner):
    last_step = None
    opts: OldOptions

    def _step(self, point: Vector, ak: float) -> tp.Tuple[Step, Vector]:
        z = self.o.f(point)
        _grad = GradientDescendRunner.grad(self.o.f, point, ak)
        res = Step(point, z)
        if self._log:
            print(res)

        def f_grad(t: float):
            return self.o.f(point - _grad * t)

        if self.last_step is None:
            self.last_step = search_up(f_grad)
        t = trinary_search(f_grad, self.last_step * 2, 10)
        self.last_step = t

        return res, point - _grad * t  # возвращаем текущий шаг и координаты для следующего
