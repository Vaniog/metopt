from utils import *


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

    def _step(self, point: Vector2D, ak: float) -> tp.Tuple[Step, Vector2D]:
        for _ in range(4):
            d = self.direction.get_v()
            next_point: Vector2D = (point[0] + d[0] * ak, point[1] + d[1] * ak)
            if self._log:
                print("point:", next_point)
                print("f:", self.p.f(*next_point))

            if self.p.f(*next_point) > self.p.f(*point):
                self.direction.rotate()
                continue

            return Step((0, 0), ak, next_point, self.p.f(*next_point)), next_point
        return Step((0, 0), ak, point, self.p.f(*point)), point

    def _run(self, start: Vector2D, a: tp.Generator, exit_condition: ExitCondition.tp) -> tp.List[Step]:
        it, next_point = self._step(start, next(a))
        steps = [it]  # тут храним все шаги программы
        while True:
            it, next_point = self._step(next_point, next(a))  # шагаем
            steps.append(it)
            if exit_condition(steps[-2], steps[-1]):  # На основании 2х последних шагов решаем, пора ли заканчивать
                break
        return steps
