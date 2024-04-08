from common.utils import *


class Direction:
    _v_i: int
    _dim = 0

    def __init__(self, dim):
        self._v_i = 0
        self._dim = dim

    def get_v(self) -> Vector:
        v_list = [0.0] * self._dim
        v_list[self._v_i // 2] = (1.0 - (self._v_i % 2 * 2))
        return Vector(*v_list)

    def rotate(self):
        self._v_i = (self._v_i + 1) % (self._dim * 2)


class CoordinateDescendRunner(OldRunner):
    dim: int
    direction: Direction
    step = 1

    def _try_step(self, point: Vector) -> tp.Tuple[Vector, bool]:
        self.dim = point.dim
        self.direction = Direction(self.dim)
        for _ in range(self.dim * 2):
            d = self.direction.get_v()
            next_point: Vector = point + (d * self.step)
            if self._log:
                print("point:", next_point)
                print("f:", self.o.f(*next_point))

            f_next_point = self.o.f(*next_point)
            if f_next_point >= self.o.f(*point):
                self.direction.rotate()
                continue
            return next_point, True
        return point, False

    def _step(self, point: Vector, ak: float) -> tp.Tuple[Step, Vector]:
        self.step = ak
        next_point, ok = self._try_step(point)
        if ok:
            return Step(next_point, self.o.f(*next_point)), next_point
        return Step(point, self.o.f(*point)), point
