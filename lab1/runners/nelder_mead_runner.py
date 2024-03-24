from scipy.optimize import minimize
import typing as tp

from lab1.runners import Step
from .utils import AbstractRunner

Vector2D = tp.Tuple[float, float]


class NelderMeadRunner(AbstractRunner):
    tol: float = 0.01

    def _step(self, point: Vector2D, ak: float) -> tp.Tuple[Step, Vector2D]:
        def f(p):
            return self.o.f(*p)

        res = minimize(f, point, method='Nelder-Mead', tol=self.tol)
        return Step(point, self.o.f(*point)), res["x"]
