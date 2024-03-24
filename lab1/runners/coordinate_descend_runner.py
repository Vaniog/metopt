from .utils import *


class Direction:
    _v_i: int
    FORWARD_X: Vector = Vector(1, 0)
    FORWARD_Y: Vector = Vector(0, 1)
    BACKWARD_X: Vector = Vector(-1, 0)
    BACKWARD_Y: Vector = Vector(0, -1)
    CYCLE: list[Vector] = [FORWARD_X, FORWARD_Y, BACKWARD_X, BACKWARD_Y]

    def __init__(self):
        self._v_i = 0

    def get_v(self) -> Vector:
        return Direction.CYCLE[self._v_i]

    def rotate(self):
        self._v_i = (self._v_i + 1) % len(Direction.CYCLE)


class CoordinateDescendRunner(AbstractRunner):
    direction: Direction = Direction()
    step = 1

    def _try_step(self, point: Vector) -> tp.Tuple[Vector, bool]:
        for _ in range(4):
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
        return Vector(0, 0), False

    def _step(self, point: Vector, ak: float) -> tp.Tuple[Step, Vector]:
        self.step = ak
        next_point, ok = self._try_step(point)
        if ok:
            return Step(next_point, self.o.f(*next_point)), next_point
        return Step(point, self.o.f(*point)), point
