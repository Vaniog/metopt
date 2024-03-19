import abc
import math
import typing as tp
from timeit import default_timer as timer
from dataclasses import dataclass

Vector2D = tp.Tuple[float, float]


@dataclass
class Step:
    """
    Класс, отвечающий за хранение отдельной итерации

    """
    grad: Vector2D
    ak: float
    point: Vector2D
    z: float

    def geogebra(self) -> str:
        """
        Форматируем под копипасту в геогебру

        """
        return f"({self.point[0]:.4f}, {self.point[1]:.4f}, {self.z:.4f})"


@dataclass
class Result:
    """
    Класс-обертка над списком итераций для удобства работы с ним
    """
    steps: tp.List[Step]

    def geogebra(self, cnt=15) -> str:
        """
        Форматируем под список точек для геогебры

        :param cnt: кол-во точек на выходе
        """
        arr = []
        step = int(len(self.steps) / cnt) + 1
        for i, el in enumerate(self.steps):
            if i % step == 0:
                arr.append(el)
        return "{" + ", ".join(map(lambda el: el.geogebra(), arr)) + "}"

    def accuracy(self, target: Vector2D):
        return Metric.EUCLID(*self.steps[-1].point, *target)


class Metric:
    """
    Тут храним разные нормы
    """
    tp = tp.Callable[[float, float, float, float], float]

    @staticmethod
    def EUCLID(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class ExitCondition:
    """
    Условие выхода - функция принимающая 2 последних шага и решающая, пора ли останавливаться
    Тут храним различные её реализации
    """
    tp = tp.Callable[[Step, Step], bool]

    @staticmethod
    def DELTA(ep: float) -> tp:
        def check(it1: Step, it2: Step):
            return abs(it1.z - it2.z) < ep

        return check

    @staticmethod
    def NORM(metric: Metric.tp, ep: float):
        return lambda it1, it2: metric(it1.point[0], it1.point[1], it2.point[0], it2.point[1]) < ep


class Coef:
    """
    Коэффициент а - в нашем случае генератор, можно было использовать не только константу, но и менять его по ходу выполнения
    """

    @staticmethod
    def GEOMETRIC_PROGRESSION(start, q):
        val = start
        while True:
            yield val
            val *= q

    @staticmethod
    def CONST(const):
        return Coef.GEOMETRIC_PROGRESSION(const, 1)


@dataclass
class Problem:
    """
    Тут храним все исходные данные, в том числе искомую точку, чтобы сравнивать полученный результат с ней
    """
    f: tp.Callable[[float, float], float]
    grad: tp.Callable[[float, float], Vector2D]
    target: Vector2D


@dataclass
class AbstractRunner(abc.ABC):
    """
    Класс, используемый для запуска программы.
    Хранит в себе все исходные данные и позволяет задавать параметры программы
    """
    p: Problem

    start: Vector2D
    a: tp.Generator
    exit_condition: ExitCondition.tp

    _log: bool = False

    @abc.abstractmethod
    def _run(self, start: Vector2D, a: tp.Generator, exit_condition: ExitCondition.tp) -> tp.List[Step]:
        raise NotImplementedError()

    def run(self) -> tp.Tuple[Result, float]:
        st = timer()
        res = Result(self._run(self.start, self.a, self.exit_condition))
        end = timer()
        return res, end - st

    def experiment(self, log=False, points=None):
        """

        :param log: если True, будут выводиться все шаги по ходу выполнения программы
        :param points: кол-во точек для геогебры
        """
        self._log = log
        res, time = self.run()
        acc = res.accuracy(self.p.target)
        print(f"Точность: {acc:.8f}")  # расстояние между полученной и искомой точками
        print(f"Кол-во шагов: {len(res.steps)}")
        print(f"Время: {time:.4f} с")
        if points:
            print(res.geogebra(points))


class GradientDescendRunner(AbstractRunner):
    def _step(self, point: Vector2D, ak: float) -> tp.Tuple[Step, Vector2D]:
        x, y = point
        z = self.p.f(*point)
        dx, dy = _grad = self.p.grad(*point)
        res = Step(_grad, ak, point, z)
        if self._log:
            print(res)
        return res, (x + ak * dx, y + ak * dy)  # возвращаем текущий шаг и координаты для следующего

    def _run(self, start: Vector2D, a: tp.Generator, exit_condition: ExitCondition.tp) -> tp.List[Step]:
        it, next_point = self._step(start, next(a))
        steps = [it]  # тут храним все шаги программы
        while True:
            it, next_point = self._step(next_point, next(a))  # шагаем
            steps.append(it)
            if exit_condition(steps[-2], steps[-1]):  # На основании 2х последних шагов решаем, пора ли заканчивать
                break
        return steps


if __name__ == '__main__':
    def f(x: float, y: float) -> float:
        return x ** 3 * y ** 5 * (4 - x - 7 * y)


    def grad(x: float, y: float) -> Vector2D:
        return (
            -x ** 3 * y ** 5 + 3 * x ** 2 * (4 - x - 7 * y) * y ** 5,
            5 * x ** 3 * (4 - x - 7 * y) * y ** 4 - 7 * x ** 3 * y ** 5
        )


    TARGET = (4 / 3, 20 / 63)
    PROBLEM = Problem(f, grad, TARGET)
    print(f(*TARGET))
    runner = GradientDescendRunner(PROBLEM, (2, 1), Coef.CONST(0.0001),
                                   ExitCondition.NORM(Metric.EUCLID, 0.0001))
    runner.experiment(False, 25)
