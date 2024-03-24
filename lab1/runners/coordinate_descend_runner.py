from .utils import *


class Direction:
    _v_i: int
    FORWARD_X: Vector2D = (1, 0)
    FORWARD_Y: Vector2D = (0, 1)
    BACKWARD_X: Vector2D = (-1, 0)
    BACKWARD_Y: Vector2D = (0, -1)
    CYCLE: list[Vector2D] = [FORWARD_X, FORWARD_Y, BACKWARD_X, BACKWARD_Y]

    def __init__(self):
        self._v_i = 0

    def get_v(self):
        return Direction.CYCLE[self._v_i]

    def rotate(self):
        self._v_i = (self._v_i + 1) % len(Direction.CYCLE)


class CoordinateDescendRunner(AbstractRunner):
    direction: Direction = Direction()
    step = 1
    step_min = 0.000001

    def _try_step(self, point: Vector2D) -> tp.Tuple[Vector2D, bool]:
        for _ in range(4):
            d = self.direction.get_v()
            next_point: Vector2D = (point[0] + d[0] * self.step, point[1] + d[1] * self.step)
            if self._log:
                print("point:", next_point)
                print("f:", self.o.f(*next_point))

            f_next_point = self.o.f(*next_point)
            if f_next_point >= self.o.f(*point):
                self.direction.rotate()
                continue
            return next_point, True
        return (0, 0), False

    def _step(self, point: Vector2D, ak: float) -> tp.Tuple[Step, Vector2D]:
        while self.step > self.step_min:
            next_point, ok = self._try_step(point)
            if ok:
                return Step(next_point, self.o.f(*next_point)), next_point
            self.step /= 2
        return Step(point, self.o.f(*point)), point
