from .utils import *
from scipy.optimize import minimize
import typing as tp
from .gradient_descent_runner import GradientDescendRunner


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


class CoordinateGradientRunner(AbstractRunner):
    def _step(self, point: Vector2D, ak: float) -> tp.Tuple[Step, Vector2D]:
        x, y = point
        z = self.p.f(*point)
        dx, dy = _grad = GradientDescendRunner.grad(self.p.f, point, ak)
        res = Step(_grad, ak, point, z)
        if self._log:
            print(res)

        def f_grad(t: float):
            return self.p.f(x - t * dx, y - t * dy)

        t = trinary_search(f_grad, ak * 1000, 10)

        return res, (x - t * dx, y - t * dy)  # возвращаем текущий шаг и координаты для следующего
