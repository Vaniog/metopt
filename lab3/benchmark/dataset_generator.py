import dataclasses
import functools
import math
import random

import numpy as np
import typing as tp
import matplotlib.pyplot as plt


class Functions:
    LINEAR = [
        lambda x: x,
        lambda x: 1
    ]
    ALL = [
        np.cos, np.sin,
        *[(lambda x: x ** p) for p in range(2)],
        *LINEAR
    ]


@dataclasses.dataclass
class GeneratorConfig:
    dim: int = 2
    rows: int = 10_000
    pow: int = 1
    noize: int = 0.01
    dist: float = 5
    functions: tp.List[tp.Callable[[float], float]] = dataclasses.field(default_factory=lambda: Functions.ALL)


@dataclasses.dataclass
class Row:
    x: np.ndarray
    y: float


def generate(output: str, cfg: GeneratorConfig):
    batch_size = 10_000
    buffer = [None] * batch_size
    noize = np.random.uniform(-cfg.noize, cfg.noize, (cfg.rows,))
    raw_func = _generate_function(cfg.dim, cfg.functions)
    i = 0
    while i < cfg.rows:
        idx = i % batch_size
        if idx == 0:
            _save_buffer(output, buffer)
        x = (np.random.random(cfg.dim) - 0.5)
        buffer[idx] = [*x, raw_func(x * cfg.dist) + noize[idx]]
        i += 1
    _save_buffer(output, buffer)
    return raw_func


def _save_buffer(path: str, buffer: tp.List[tp.List[float]]):
    with open(path, 'a') as f:
        lines = []
        for row in filter(lambda el: el is not None, buffer):
            lines.append(",".join(map(str, row)) + "\n")
        f.writelines(lines)


def _generate_function(dim: int, functions: list[tp.Callable[[float], float]]) -> tp.Callable[[np.ndarray], float]:
    border = 5
    coefs = [(np.random.random(len(functions)) - 0.5) * border for _ in range(dim)]

    def inner(vec: np.ndarray) -> int:
        res = 0
        for var in range(dim):
            for i, f in enumerate(functions):
                res += f(vec[var]) * coefs[var][i]
        return res

    return inner


def plot3d(objective: tp.Callable):
    # define range for input
    r_min, r_max = -10.0, 10.0
    # sample input range uniformly at 0.1 increments
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis)
    # compute targets
    results = objective([x, y])
    # create a surface plot with the jet color scheme
    figure = plt.figure()
    axis = plt.axes(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    axis.set_title(objective.__name__)
    # show the plot
    plt.show()


def plot2d(objective: tp.Callable):
    # Определяем диапазон для входных значений
    r_min, r_max = -10.0, 10.0
    # Генерируем входные значения равномерно с шагом 0.1
    xaxis = np.arange(r_min, r_max, 0.1)
    # Вычисляем значения функции для каждой точки сетки
    results = objective([xaxis])
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
    # plot3d(_generate_function(2, Functions.LINEAR))
    # plot2d(_generate_function(1, [*Functions.LINEAR, lambda x: x ** 2]))
    generate("test.csv", GeneratorConfig())
