import abc
import dataclasses
import math
import typing as tp
from dataclasses import dataclass
from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt

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

        step = int(len(self.steps) / cnt) + 1
        arr = self.steps[::step]

        return "{" + ", ".join(map(lambda el: el.geogebra(), arr)) + "}"

    def accuracy(self, target: Vector2D):
        return Metric.EUCLID(*self.steps[-1].point, *target)

    def plot(self, ax: plt.Axes, cnt=15):
        xdata = []
        ydata = []
        zdata = []

        for step in self._steps(cnt):
            xdata.append(step.point[0])
            ydata.append(step.point[1])
            zdata.append(step.z)

        ax.scatter(xdata, ydata, zdata)

    def _steps(self, cnt: int):
        step = int(len(self.steps) / cnt) + 1
        return self.steps[::step]

    def calculate_scale(self, cnt):
        return tuple(sorted([self._steps(cnt)[-1].point[0], self._steps(cnt)[0].point[1]]))


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
    def NORM(metric: Metric.tp, ep: float) -> tp:
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
    target: Vector2D


@dataclass
class PlotConfig:
    linspace_start: float
    linspace_stop: float
    linspace_num: int = 30
    func_num: int = 150
    dpi: int = 1000
    steps: int = 10

    def copy(self, xs, ys):
        return PlotConfig(xs, ys, **dataclasses.asdict(self))


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
    def _step(self, point: Vector2D, ak: float) -> tp.Tuple[Step, Vector2D]:
        raise NotImplementedError()

    def _run(self, start: Vector2D, a: tp.Generator, exit_condition: ExitCondition.tp) -> tp.List[Step]:
        it, next_point = self._step(start, next(a))
        steps = [it]  # тут храним все шаги программы
        while True:
            it, next_point = self._step(next_point, next(a))  # шагаем
            steps.append(it)
            if exit_condition(steps[-2], steps[-1]):  # На основании 2х последних шагов решаем, пора ли заканчивать
                break
        return steps

    def run(self) -> tp.Tuple[Result, float]:
        st = timer()
        res = Result(self._run(self.start, self.a, self.exit_condition))
        end = timer()
        return res, end - st

    def experiment(self, log=False, points=None, plt_cfg: PlotConfig = None):
        """

        :param log: если True, будут выводиться все шаги по ходу выполнения программы
        :param points: кол-во точек для геогебры
        :param plt_cfg: конфигурация графика
        """
        self._log = log
        res, time = self.run()
        acc = res.accuracy(self.p.target)
        print(f"Точность: {acc:.8f}")  # расстояние между полученной и искомой точками
        print(f"Кол-во шагов: {len(res.steps)}")
        print(f"Время: {time:.4f} с")
        if points:
            print(res.geogebra(points))
            print(res.steps[len(res.steps) - 1].point)

        self.func_plot(plt_cfg)

        self.result_plot(plt_cfg, points, res)

    def result_plot(self, plt_cfg, points, res):
        fig = plt.figure(dpi=plt_cfg.dpi)
        ax = plt.axes(projection='3d')
        plt_cfg.linspace_start, plt_cfg.linspace_stop = res.calculate_scale(points)
        self.plot(ax, plt_cfg)
        res.plot(ax, points)
        fig.show()

    def func_plot(self, plt_cfg):
        fig = plt.figure(dpi=plt_cfg.dpi)
        ax = plt.axes(projection='3d')
        self.plot(ax, plt_cfg)
        fig.show()

    def plot(self, ax: plt.Axes, cfg: PlotConfig):
        x = np.linspace(cfg.linspace_start, cfg.linspace_stop, cfg.linspace_num)
        y = np.linspace(cfg.linspace_start, cfg.linspace_stop, cfg.linspace_num)

        X, Y = np.meshgrid(x, y)
        Z = self.p.f(X, Y)

        ax.contour3D(X, Y, Z, cfg.func_num, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
