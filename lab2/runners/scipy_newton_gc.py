from scipy import optimize

from common.utils import AbstractRunner, Vector, Oracle


class ScipyNewtonGCRunner(AbstractRunner):
    def _run(self, start: Vector, *args, **kwargs):
        optimize.minimize(self.o.f, start.ndarray(), method="L-BFGS-B")


if __name__ == '__main__':
    def f(x, y):
        return x ** 2 + y ** 2


    o = Oracle(f, Vector(0, 0))
    r = ScipyNewtonGCRunner(o, Vector(15, 15))
    r.experiment()
