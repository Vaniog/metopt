from .utils import *


class GradientDescendRunner(AbstractRunner):
    @staticmethod
    def grad(f: tp.Callable[[float, float], float], p: Vector2D, delta: float) -> Vector2D:
        dfx = f(p[0] + delta, p[1]) - f(*p)  # изменение f по x
        dfy = f(p[0], p[1] + delta) - f(*p)  # изменение f по y
        # градиент, составленный из частных производных
        return dfx / delta, dfy / delta

    def _step(self, point: Vector2D, ak: float) -> tp.Tuple[Step, Vector2D]:
        x, y = point
        z = self.p.f(*point)
        dx, dy = _grad = self.grad(self.p.f, point, ak)
        res = Step(point, z)
        if self._log:
            print(res)
        return res, (x - ak * dx, y - ak * dy)  # возвращаем текущий шаг и координаты для следующего
