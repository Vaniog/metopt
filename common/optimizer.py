import dataclasses
import random
import typing as tp
from itertools import chain
from statistics import mean

import numpy as np
from scipy.optimize import minimize

from common.benchmark import BenchmarkResult, MAX_INT
from common.functions import custom
from common.functions.functions import functions
from common.utils import AbstractRunner, Options, Oracle, Vector, plot
from lab1.runners import CoordinateDescendImprovedRunner
from lab2.runners import WolfeOptions, NewtonConstRunner, NewtonConstOptions, NewtonSearchRunner, WolfeRunner


class ErrorFunction:
    _IN = tp.Iterable[BenchmarkResult.ExperimentResult]
    T = tp.Callable[[_IN], float]

    @staticmethod
    def accuracy(res: _IN):
        return mean(map(lambda el: el.accuracy, res))


def construct_func(runner_type: tp.Type[AbstractRunner],
                   error_function: ErrorFunction.T, ):
    opts_type = runner_type.opts_type()
    fields = _fields(opts_type)

    def construct_options(args):
        if len(args) != len(fields):
            raise Exception(f"args cnt {len(args)} != fields cnt {len(fields)}")
        res = {}
        for i, (n, t) in enumerate(fields.items()):
            res[n] = t(args[i])

        return opts_type(**res)

    def f(*args):
        if len(args) == 1:
            args = args[0]
        options = construct_options(args)
        print("attempt")
        print(f"--{options}")
        if not options.validate():
            print("incorrect options")
            return MAX_INT
        res = []
        for func in custom():
            res.append(_attempt(runner_type, func, options))
        error = error_function(chain(*res))
        print(f"--error: {error}")
        return error

    return f, construct_options


def optimize_params(runner_type: tp.Type[AbstractRunner],
                    error_function: ErrorFunction.T,
                    start: Options | None = None):
    opts_type = runner_type.opts_type()
    print("#" * 5 + "OPTIMIZE" + "#" * 5)
    print(f"optimizing cls {runner_type.__name__}")
    fields = _fields(opts_type)
    if not start:
        start = opts_type.default()
    print(f"options list: ({', '.join(fields.keys())})")
    print(f"starting from {start}")
    print("#" * 6 + "START" + "#" * 7)
    print()

    f, construct_options = construct_func(runner_type, error_function)

    def arr_from_opts(opts: Options):
        return np.array([getattr(opts, k) for k in fields.keys()])

    # m = minimize(f, arr_from_opts(start))
    # print(m)
    # return construct_options(m["x"])
    r = CoordinateDescendImprovedRunner(Oracle(f, None), Vector(*arr_from_opts(start)), None)
    res = r.run()[0].steps[-1]
    print(res)
    return construct_options(res.point.coords)


def _fields(cls):
    res = {}
    for filed in dataclasses.fields(cls):
        if filed.metadata.get("fixed"):
            continue
        res[filed.name] = filed.type

    return res


def _attempt(runner: tp.Type[AbstractRunner], func, opts: Options, attempts=10):
    res = []
    for t in range(attempts):
        start = Vector(*[random.randint(-10, 10)] * func.dim)
        oracle = Oracle(func, func.target())
        b = BenchmarkResult.process_runner(runner(oracle, start, opts))
        res.append(b)
    return res


if __name__ == '__main__':
    m = optimize_params(
        WolfeRunner,
        ErrorFunction.accuracy,
    )
    print(m)
#     NewtonConstOptions(exit_condition_threshold=0.375, learning_rate=0.49609375, grad_delta=0.5)
# NewtonConstOptions(exit_condition_threshold=0.1349847412109375, learning_rate=0.875, grad_delta=0.001)
# NewtonConstOptions(exit_condition_threshold=0.01, learning_rate=1.2498779296875, grad_delta=0.001)
