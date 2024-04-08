from scipy import optimize

from common.utils import AbstractRunner, Vector, Oracle, PlotConfig
from lab2.runners.newton_const import NewtonConstRunner, NewtonConstOptions


class ScipyNewtonGCRunner(AbstractRunner):
    def _run(self, start: Vector, *args, **kwargs):
        optimize.minimize(self.o.f, start.ndarray(), method="L-BFGS-B", jac=lambda x: optimize.approx_fprime(x, self.o.f, 0.01))


if __name__ == '__main__':
    def f(x, y):
        return x ** 2 + y ** 2

    # j = grad(f)
    # print(j([0, 1]))
    r = ScipyNewtonGCRunner(Oracle(f, Vector(0, 0)), Vector(15, 15))
    r2 = NewtonConstRunner(Oracle(f, Vector(0, 0)), Vector(15, 15), NewtonConstOptions())
    r.experiment(plt_cfg=PlotConfig(draw_steps=False))
    r2.experiment(plt_cfg=PlotConfig(draw_steps=False))
