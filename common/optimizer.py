import dataclasses
import random
import typing as tp
from itertools import chain
from statistics import mean

import numpy as np
from scipy.optimize import minimize

from common.benchmark import BenchmarkResult
from common.functions.functions import functions
from common.utils import AbstractRunner, Options, Oracle, Vector
from lab1.runners import CoordinateDescendImprovedRunner
from lab2.runners import WolfeOptions, NewtonConstRunner, NewtonConstOptions, NewtonSearchRunner, WolfeRunner


class ErrorFunction:
    _IN = tp.Iterable[BenchmarkResult.ExperimentResult]
    T = tp.Callable[[_IN], float]

    @staticmethod
    def accuracy(res: _IN):
        return mean(map(lambda el: el.accuracy, res))


def optimize_params(runner_type: tp.Type[AbstractRunner],
                    error_function: ErrorFunction.T):
    opts_type = runner_type.opts_type()
    print("#" * 5 + "OPTIMIZE" + "#" * 5)
    print(f"optimizing cls {runner_type.__name__}")
    fields = _fields(opts_type)
    print(f"options list: ({', '.join(fields.keys())})")
    print("#" * 6 + "START" + "#" * 7)
    print()

    def construct_options(args):
        if len(args) != len(fields):
            raise Exception(f"args cnt {len(args)} != fields cnt {len(fields)}")
        res = {}
        for i, (n, t) in enumerate(fields.items()):
            res[n] = t(args[i])

        return opts_type(**res)

    def f(*args):
        options = construct_options(args)
        print("attempt")
        print(f"--{options}")
        res = []
        for func in functions():
            res.append(_attempt(runner_type, func, options))
        error = error_function(chain(*res))
        print(f"--error: {error}")
        return error

    # return minimize(f, np.ones(len(fields)), method="Nelder-Mead")
    r = CoordinateDescendImprovedRunner(Oracle(f, None), Vector(*([1] * len(fields))), None)
    return r.run()[0].steps[-1]


def _fields(cls):
    res = {}
    for filed in dataclasses.fields(cls):
        if not filed.repr:
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
        NewtonConstRunner,
        ErrorFunction.accuracy,
    )
    print(m)
