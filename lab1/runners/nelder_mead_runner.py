from scipy.optimize import minimize
import typing as tp

from common.utils import Step, Vector, OldRunner, OldOptions


class NelderMeadRunner(OldRunner):
    tol: float = 0.01
    opts: OldOptions

    def _step(self, point: Vector, ak: float) -> tp.Tuple[Step, Vector]:
        def f(p):
            return self.o.f(*p)

        res = minimize(f, point.coords, method='Nelder-Mead', tol=self.tol)
        return Step(point, self.o.f(*point)), Vector(*res["x"])
