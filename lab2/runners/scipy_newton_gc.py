from dataclasses import dataclass

from scipy import optimize

from common.utils import AbstractRunner, Vector, Oracle, PlotConfig, Coef, ExitCondition, Metric, Options
from lab2.runners.newton_const import NewtonConstRunner, NewtonConstOptions


@dataclass
class ScipyNewtonCGOptions(Options):
    pass


class ScipyNewtonCGRunner(AbstractRunner):
    opts: ScipyNewtonCGOptions

    def _run(self, start: Vector, *args, **kwargs):
        optimize.minimize(self.o.f, start.ndarray(), method="Newton-CG",
                          jac=lambda x: optimize.approx_fprime(x, self.o.f, 0.01),
                          tol=self.opts.exit_condition_threshold
                          )


class ScipyQuasiNewtonRunner(AbstractRunner):
    opts: ScipyNewtonCGOptions
    
    def _run(self, start: Vector, *args, **kwargs):
        optimize.minimize(self.o.f, start.ndarray(), method="BFGS",
                          jac=lambda x: optimize.approx_fprime(x, self.o.f, 0.01),
                          tol=self.opts.exit_condition_threshold
                          )

# if __name__ == '__main__':
#     def f(x, y):
#         return x ** 2 + y ** 10
#
#
#     start = Vector(15, 1)
#     r = ScipyNewtonCGRunner(Oracle(f, Vector(0, 0)), start)
#     r2 = NewtonConstRunner(Oracle(f, Vector(0, 0)), start, NewtonConstOptions())
#     r3 = CoordinateDescendImprovedRunner(Oracle(f, Vector(0, 0)), start, Coef.GEOMETRIC_PROGRESSION(0.001, 0.999999),
#                                          ExitCondition.NORM(Metric.EUCLID, 0.00001))
#     r.experiment(plt_cfg=PlotConfig(draw_steps=False))
#     r2.experiment(plt_cfg=PlotConfig(draw_steps=False))
#     r3.experiment(plt_cfg=PlotConfig(draw_steps=False))
