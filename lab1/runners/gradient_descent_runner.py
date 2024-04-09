from common.utils import *


class GradientDescendRunner(OldRunner):
    opts: OldOptions
    @staticmethod
    def grad(f: tp.Callable[[Vector], float], p: Vector, delta: float) -> Vector:
        ds = []
        for i, el in enumerate(p):
            coords = [*p]
            coords[i] += delta
            ds.append(f(Vector(*coords)) - f(*p))

        # градиент, составленный из частных производных
        res = Vector(*map(lambda el: el / delta, ds))
        # dfx = f(p[0] + delta, p[1]) - f(*p)  # изменение f по x
        # dfy = f(p[0], p[1] + delta) - f(*p)  # изменение f по y
        # res = Vector()
        # print(res)
        return res

    def _step(self, point: Vector, ak: float) -> tp.Tuple[Step, Vector]:
        z = self.o.f(*point)
        _grad = self.grad(self.o.f, point, ak)
        res = Step(point, z)
        if self._log:
            print(res)
        return res, point - Vector(*map(lambda el: el * ak, _grad))  # возвращаем текущий шаг и координаты для следующего
