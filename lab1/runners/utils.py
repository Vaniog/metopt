import abc
import dataclasses
import math
import typing as tp
from dataclasses import dataclass
from functools import wraps, lru_cache
from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mtl

Vector2D = tp.Tuple[float, float]


@dataclass
class Step:
    """
    Класс, отвечающий за хранение отдельной итерации

    """
    point: Vector2D
    z: float

    def geogebra(self) -> str:
        """
        Форматируем под копипасту в геогебру

        """
        return f"({self.point[0]:.4f}, {self.point[1]:.4f}, {self.z:.4f})"

    def __hash__(self):
        return hash(self.point) + hash(self.z)


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

    def plot(self, ax: plt.Axes, cnt=15, flat=False):
        xdata = []
        ydata = []
        zdata = []
        color = []

        for i, step in enumerate(self.filtered_steps(cnt)):
            xdata.append(step.point[0])
            ydata.append(step.point[1])
            zdata.append(step.z)
            color.append(math.pow(i, 1 / 4))

        if flat:
            ax.scatter(xdata, ydata, c=color, cmap='afmhot')
        else:
            ax.scatter(xdata, ydata, zdata, c=color, cmap='afmhot')

    @lru_cache
    def filtered_steps(self, cnt: int):
        step = int(len(self.steps) / cnt) + 1
        return self.steps[::step]

    def __hash__(self):
        return hash(self.steps[-1])


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
class Oracle:
    """
    Тут храним все исходные данные, в том числе искомую точку, чтобы сравнивать полученный результат с ней
    """
    f: tp.Callable[[float, float], float]
    target: Vector2D
    steps: list[Step] = dataclasses.field(default_factory=list)

    def dec(self, f: tp.Callable):
        @wraps(f)
        def inner(x: float, y: float) -> float:
            r = f(x, y)
            self.steps.append(Step((x, y), r))
            return r

        return inner

    def __post_init__(self):
        self.f = self.dec(self.f)


@dataclass
class PlotConfig:
    linspace_start: float
    linspace_stop: float
    linspace_num: int = 30
    func_num: int = 150
    dpi: int = 1000
    steps: int = 10
    level_lines: int = 15
    scale_coef = 0.25

    _x_start: float = -1
    _x_stop: float = -1
    _y_start: float = -1
    _y_stop: float = -1

    def copy(self, xs, ys):
        return PlotConfig(xs, ys, **dataclasses.asdict(self))

    @staticmethod
    def __get_coordinate(idx: int):
        return lambda el: el.point[idx]

    def _calculate_scale(self, idx: int, res: Result, cnt: int):
        start, stop = (
            min(map(self.__get_coordinate(idx), res.filtered_steps(cnt))),
            max(map(self.__get_coordinate(idx), res.filtered_steps(cnt)))
        )
        l = stop - start

        return start - (l * self.scale_coef / 2), stop + (l * self.scale_coef / 2)

    def calculate_scale(self, res: Result, cnt: int):
        self._x_start, self._x_stop = self._calculate_scale(0, res, cnt)
        self._y_start, self._y_stop = self._calculate_scale(1, res, cnt)

    def __post_init__(self):
        self._x_start = self._y_start = self.linspace_start
        self._x_stop = self._y_stop = self.linspace_stop


@dataclass
class AbstractRunner(abc.ABC):
    """
    Класс, используемый для запуска программы.
    Хранит в себе все исходные данные и позволяет задавать параметры программы
    """
    o: Oracle

    start: Vector2D
    a: tp.Generator
    exit_condition: ExitCondition.tp

    _log: bool = False

    @abc.abstractmethod
    def _step(self, point: Vector2D, ak: float) -> tp.Tuple[Step, Vector2D]:
        raise NotImplementedError()

    def _run(self, start: Vector2D, a: tp.Generator, exit_condition: ExitCondition.tp):
        it, next_point = self._step(start, next(a))
        steps = [it]  # тут храним все шаги программы
        while True:
            it, next_point = self._step(next_point, next(a))  # шагаем
            steps.append(it)
            if exit_condition(steps[-2], steps[-1]):  # На основании 2х последних шагов решаем, пора ли заканчивать
                break

    def run(self) -> tp.Tuple[Result, float]:
        st = timer()
        self._run(self.start, self.a, self.exit_condition)
        end = timer()

        res = Result(self.o.steps)
        self.o.steps = []
        return res, end - st

    def experiment(self, log=False, points=None, plt_cfg: PlotConfig = None):
        """

        :param log: если True, будут выводиться все шаги по ходу выполнения программы
        :param points: кол-во точек для геогебры
        :param plt_cfg: конфигурация графика
        """
        self._log = log
        res, time = self.run()
        acc = res.accuracy(self.o.target)
        print(f"Точность (расстояние до реального минимума): {acc:.8f}")  # расстояние между полученной и искомой точками
        print(f"Кол-во запросов к оракулу: {len(res.steps)}")
        print(f"Время: {time:.4f} с")
        if points:
            points = min(points, len(res.steps))
            # print(res.geogebra(points))
            # print(res.steps[len(res.steps) - 1].point)

        # self.func_plot(plt_cfg)
        # self.level_curves(plt_cfg)

        plt_cfg.calculate_scale(res, points)
        self.result_plot(plt_cfg, points, res)
        self.res_level_curves(plt_cfg, points, res)

    def result_plot(self, plt_cfg, points, res):
        fig = plt.figure(dpi=plt_cfg.dpi)
        ax = plt.axes(projection='3d')
        self.plot(ax, plt_cfg)
        res.plot(ax, points)
        fig.show()

    def func_plot(self, plt_cfg):
        fig = plt.figure(dpi=plt_cfg.dpi)
        ax = plt.axes(projection='3d')
        self.plot(ax, plt_cfg)
        fig.show()

    def level_curves(self, cfg: PlotConfig):
        fig = plt.figure(dpi=cfg.dpi)
        ax = plt.axes()
        self._level_curves(ax, cfg)
        fig.show()

    def res_level_curves(self, cfg: PlotConfig, cnt: int, res: Result):
        fig = plt.figure(dpi=cfg.dpi)
        ax = plt.axes()
        res.plot(ax, cnt, flat=True)
        self._level_curves(ax, cfg)

        fig.show()

    def _level_curves(self, ax, cfg: PlotConfig):
        X, Y = np.meshgrid(*self._linspace(cfg))
        Z = self.o.f(X, Y)
        ax.contour(X, Y, Z, levels=cfg.level_lines, cmap='viridis')

    @staticmethod
    def _linspace(cfg):
        return (
            np.linspace(cfg._x_start, cfg._x_stop, cfg.linspace_num),
            np.linspace(cfg._y_start, cfg._y_stop, cfg.linspace_num)
        )

    def plot(self, ax: plt.Axes, cfg: PlotConfig):
        X, Y = np.meshgrid(*self._linspace(cfg))
        Z = self.o.f(X, Y)

        ax.contour3D(X, Y, Z, cfg.func_num, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
