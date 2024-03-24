from .utils import *
from .coordinate_descend_runner import Direction


class CoordinateDescendImprovedRunner(AbstractRunner):
    direction: Direction
    step = 1
    step_min = 0.000001

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
        while self.step > self.step_min:
            next_point, ok = self._try_step(point)
            if ok:
                return Step(next_point, self.o.f(*next_point)), next_point
            self.step /= 2
        return Step(point, self.o.f(*point)), point
