import abc
import dataclasses
import inspect
import math
import signal
import typing as tp
from abc import ABC
from dataclasses import dataclass
import time
from functools import wraps, lru_cache
import platform
from timeit import default_timer as timer
from typing import Iterator

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mtl

from overtake import overtake
from typing_extensions import overload


# class _Running:
#     __timeout: float
#     __start_time: float
#
#     def __init__(self):
#         self.__timeout = -1
#         self.__start_time = -1
#
#     def start(self):
#         if self.__start_time != -1:
#             raise Exception("Another instance is running")
#         self.__start_time = timer()
#
#     def stop(self):
#         res = timer() - self.__start_time
#         self.__start_time = -1
#         return res
#
#     def set_timeout(self, t):
#         self.__timeout = t
#
#     def __call__(self, *args, **kwargs):
#         if timer() - self.__start_time > self.__timeout:
#             print("TIMEOUT")
#             raise TimeoutError()
#         return True
#
#
# running = _Running()


class Vector(tp.Iterable):
    coords: tp.Tuple[float]
    dim: int

    def __init__(self, *coords: float):
        self.coords = coords

    @property
    def dim(self) -> int:
        return len(self.coords)

    def __getitem__(self, item):
        return self.coords[item]

    def ndarray(self) -> np.ndarray:
        return np.array(self.coords)

    def assert_equal_dim(self, other):
        if self.dim != other.dim:
            raise Exception(
                f"unable to perform operation on vectors with dim {self.dim} and {other.dim}")

    def __op(self, other, operation: tp.Callable):
        if not isinstance(other, Vector):
            raise Exception(
                f"can only be applied to another {self.__class__.__name__}")
        self.assert_equal_dim(other)
        return Vector(*map(lambda it: operation(it[1], other[it[0]]), enumerate(self.coords)))

    def __add__(self, other):
        return self.__op(other, lambda el1, el2: el1 + el2)

    def __sub__(self, other):
        return self.__op(other, lambda el1, el2: el1 - el2)

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise Exception("can only be applied to vector and number")
        return Vector(*map(lambda el: el * other, self))

    def __iter__(self) -> Iterator:
        return iter(self.coords)

    def __repr__(self):
        return f"V{str(self.coords)}"


@dataclass
class Step:
    """
    Класс, отвечающий за хранение отдельной итерации

    """
    point: Vector
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

    def queries(self):
        return len(self.steps)

    def accuracy(self, target: Vector | None):
        if target is None:
            return None
        return Metric.EUCLID(self.steps[-1].point, target)

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
            ax.scatter(xdata, ydata, c=color, cmap='afmhot', zorder=2)
        else:
            ax.scatter(xdata, ydata, zdata, c=color, cmap='afmhot', zorder=2)

    @lru_cache
    def filtered_steps(self, cnt: int):
        step = int(len(self.steps) / cnt) + 1
        return self.steps[::step]

    def __hash__(self):
        return hash(self.steps[-1])


def _check_types(tp: type, *args):
    return all(map(lambda v: isinstance(v, tp), args))


class Metric:
    """
    Тут храним разные нормы
    """
    _vector = Vector | np.ndarray
    tp = tp.Callable[[_vector, _vector], float]

    @classmethod
    def EUCLID(cls, v1: _vector, v2: _vector) -> float:
        if _check_types(Vector, v1, v2):
            v1.assert_equal_dim(v2)
            return math.sqrt(sum(map(lambda it: (it[1] - v2[it[0]]) ** 2, enumerate(v1))))
        elif _check_types(np.ndarray, v1, v2):
            return np.linalg.norm(v2 - v1)


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
        def inner(it1, it2):
            if _check_types(Step, it1, it2):
                return metric(it1.point, it2.point) < ep
            else:
                return metric(it1, it2) < ep

        return inner


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


class Oracle:
    """
    Тут храним все исходные данные, в том числе искомую точку, чтобы сравнивать полученный результат с ней
    """
    f: tp.Callable[[Vector], float]
    target: Vector | None
    steps: list[Step]

    def __init__(self, f: tp.Callable, target: Vector | None):
        self.f = self.dec(f)
        self.target = target
        self.steps = []

    def dec(self, f: tp.Callable):
        f = lru_cache(f)

        @wraps(f)
        def inner(v: Vector, *args) -> float:
            if isinstance(v, np.ndarray):
                v = Vector(*v)
            elif not isinstance(v, (Vector, list, tuple)):
                v = Vector(*((v,) + args))

            try:
                r = f(*v)
            except TypeError:
                r = f.__wrapped__(*v)
            self.steps.append(Step(v, r))
            return r

        return inner


@dataclass
class PlotConfig:
    linspace_start: float = -5
    linspace_stop: float = 5
    linspace_num: int = 30
    func_lines_num: int = 150
    dpi: int = 1000
    steps: int = 10
    level_lines: int = 15
    scale_coef: float = 0.25
    draw_function: bool = False
    draw_steps: bool = True

    _x_start = -1
    _x_stop = -1
    _y_start = -1
    _y_stop = -1

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


class RunnerMeta(abc.ABCMeta, type):
    runners = []

    def __new__(mcs, name, bases, dct):
        new_runner = super().__new__(mcs, name, bases, dct)
        if not inspect.isabstract(new_runner):
            mcs.runners.append(new_runner)
        return new_runner


@dataclass
class Options:
    exit_condition_threshold: float = dataclasses.field(default=0.001, metadata={"fixed": True})
    exit_condition: ExitCondition.tp = dataclasses.field(default=None, repr=False, metadata={"fixed": True})

    def __post_init__(self):
        if not self.exit_condition:
            self.exit_condition = ExitCondition.NORM(Metric.EUCLID, self.exit_condition_threshold)

    def validate(self):
        return 0 < self.exit_condition_threshold < 1

    @classmethod
    def default(cls, override: dict = None):
        if override:
            return cls(**override)
        return cls()

    def copy(self, **kwargs):
        d = dataclasses.asdict(self)
        d.update(kwargs)
        # noinspection PyArgumentList
        return type(self)(**d)


class AbstractRunner(abc.ABC, metaclass=RunnerMeta):
    """
    Класс, используемый для запуска программы.
    Хранит в себе все исходные данные и позволяет задавать параметры программы
    """

    o: Oracle
    start: Vector
    opts: Options

    TIMEOUT: float = 1
    __start_time: float = -1

    def __init__(self, o: Oracle, start: Vector, opts: Options | None = None, override_opts: dict | None = None):
        self.o = o
        self.start = start
        if not opts:
            opts = self.__default_opts(override_opts)
        self.opts = opts
        self._log = False

    def __default_opts(self, override: dict = None) -> Options:
        return self.opts_type().default(override)

    @classmethod
    def opts_type(cls):
        for opts_type in cls.__annotations__.values():
            if isinstance(opts_type, type) and issubclass(opts_type, Options):
                return opts_type
        raise AttributeError(f"All runners classes must have opts annotation, {cls.__name__} hasn't")

    def set_log(self, log: bool):
        self._log = log

    @abc.abstractmethod
    def _run(self, start: Vector, *args, **kwargs):
        raise NotImplemented()

    def running(self):
        if timer() - self.__start_time > self.TIMEOUT:
            raise TimeoutError("timeout")
        return True

    def run(self) -> tp.Tuple[Result, float]:
        st = timer()
        self.__start_time = st
        self._run(self.start)
        end = timer()

        res = Result(self.o.steps)
        self.o.steps = []
        return res, end - st

    def experiment(self, log=False, points=100, plt_cfg: PlotConfig = None):
        """

        :param log: если True, будут выводиться все шаги по ходу выполнения программы
        :param points: кол-во точек для графика
        :param plt_cfg: конфигурация графика
        """
        self._log = log
        if not plt_cfg:
            plt_cfg = PlotConfig()
        res, time = self.run()
        acc = res.accuracy(self.o.target)
        print(
            f"Точность (расстояние до реального минимума): {acc:.8f}")  # расстояние между полученной и искомой точками
        print(f"Кол-во запросов к оракулу: {len(res.steps)}")
        print(f"Время: {time:.4f} с")
        if points:
            points = min(points, len(res.steps))
            # print(res.geogebra(points))
            # print(res.steps[len(res.steps) - 1].point)

        dim = res.steps[-1].point.dim
        if dim != 2:
            print(f"can't draw plot for {dim}-dimensional function")
            return
        if plt_cfg.draw_function:
            self.func_plot(plt_cfg)
            self.level_curves(plt_cfg)

        if plt_cfg.draw_steps:
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
        self._level_curves(ax, cfg)
        res.plot(ax, cnt, flat=True)

        fig.show()

    def _level_curves(self, ax, cfg: PlotConfig):
        X, Y = np.meshgrid(*self._linspace(cfg))
        Z = self.o.f(Vector(X, Y))
        ax.contour(X, Y, Z, levels=cfg.level_lines, cmap='viridis')

    @staticmethod
    def _linspace(cfg):
        return (
            np.linspace(cfg._x_start, cfg._x_stop, cfg.linspace_num),
            np.linspace(cfg._y_start, cfg._y_stop, cfg.linspace_num)
        )

    def plot(self, ax: plt.Axes, cfg: PlotConfig):
        X, Y = np.meshgrid(*self._linspace(cfg))
        Z = self.o.f(Vector(X, Y))

        ax.contour3D(X, Y, Z, cfg.func_lines_num, cmap='viridis', alpha=0.4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


@dataclass
class OldOptions(Options):
    a: tp.Generator = Coef.CONST(0.001)


class OldRunner(AbstractRunner, ABC):
    a: tp.Generator
    exit_condition: ExitCondition.tp

    opts: OldOptions

    @overload
    def __init__(self, o: Oracle, start: Vector, a: tp.Generator,
                 exit_condition: ExitCondition.tp):
        super().__init__(o, start, OldOptions(exit_condition=exit_condition, a=a))
        self.a = a
        self.exit_condition = exit_condition

    @overload
    def __init__(self, o: Oracle, start: Vector, opts: OldOptions | None):
        super().__init__(o, start, opts, None)
        self.a = self.opts.a
        self.exit_condition = self.opts.exit_condition

    # noinspection PyMissingConstructor
    @overtake
    def __init__(*args, **kwargs):
        ...

    @abc.abstractmethod
    def _step(self, point: Vector, ak: float) -> tp.Tuple[Step, Vector]:
        raise NotImplementedError()

    def _run(self, start: Vector, *args, **kwargs):
        it, next_point = self._step(start, next(self.opts.a))
        steps = [it]  # тут храним все шаги программы
        while True:
            it, next_point = self._step(next_point, next(self.a))  # шагаем
            steps.append(it)
            # На основании 2х последних шагов решаем, пора ли заканчивать
            if self.exit_condition(steps[-2], steps[-1]):
                break


def plot(objective: tp.Callable[[float, float], float]):
    # define range for input
    r_min, r_max = -10.0, 10.0
    # sample input range uniformly at 0.1 increments
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis)
    # compute targets
    results = objective(x, y)
    # create a surface plot with the jet color scheme
    figure = plt.figure()
    axis = plt.axes(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    axis.set_title(objective.__name__)
    # show the plot
    plt.show()

def plot(objective: tp.Callable[[float, float], float]):
    # define range for input
    r_min, r_max = -10.0, 10.0
    # sample input range uniformly at 0.1 increments
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis)
    # compute targets
    results = objective(x, y)
    # create a surface plot with the jet color scheme
    figure = plt.figure()
    axis = plt.axes(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    axis.set_title(objective.__name__)
    # show the plot
    plt.show()


def plot_2d(objective: tp.Callable[[float], float]):
    # Определяем диапазон для входных значений
    r_min, r_max = -10.0, 10.0
    # Генерируем входные значения равномерно с шагом 0.1
    xaxis = np.arange(r_min, r_max, 0.1)
    # Вычисляем значения функции для каждой точки сетки
    results = objective(xaxis)
    # Строим обычный 2D график
    plt.plot(xaxis, results)
    # plt.colorbar(label='Значения функции')
    plt.title(objective.__name__)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    # Показываем график
    plt.show()

if __name__ == '__main__':
    def f(x):
        return np.sin(x) / x
    plot_2d(f)
