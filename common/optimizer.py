import dataclasses
import random
import typing as tp
from itertools import chain
from statistics import mean

import numpy as np
from scipy.optimize import minimize, least_squares, brute

from common.benchmark import BenchmarkResult, MAX_INT
from common.functions import custom
from common.functions import functions
from common.utils import AbstractRunner, Options, Oracle, Vector, plot, OldOptions, ExitCondition, Coef
from lab1.runners import CoordinateDescendImprovedRunner, GradientDescendRunner
from lab2.runners import WolfeOptions, NewtonConstRunner, NewtonConstOptions, NewtonSearchRunner, WolfeRunner


class ErrorFunction:
    _IN = tp.Iterable[BenchmarkResult.ExperimentResult]
    T = tp.Callable[[_IN], float]

    @staticmethod
    def accuracy(res: _IN):
        return mean(map(lambda el: el.accuracy, res))


def construct_func(runner_type: tp.Type[AbstractRunner],
                   error_function: ErrorFunction.T,
                   start: Options):
    fields = _fields(type(start))

    def construct_options(args):
        if len(args) != len(fields):
            raise Exception(f"args cnt {len(args)} != fields cnt {len(fields)}")
        res = {}
        for i, (n, t) in enumerate(sorted(fields.items())):
            res[n] = t.type(args[i])

        return start.copy(**res)

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
        for func in functions(exclude=["Abs", 'Stairs']):
            res.append(_attempt(runner_type, func, options))
        error = error_function(chain(*res))
        print(f"--error: {error}")
        return error

    return f, construct_options


def optimize_params(runner_type: tp.Type[AbstractRunner],
                    error_function: ErrorFunction.T,
                    start: Options | None = None,
                    tol: float = 0.1):
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

    f, construct_options = construct_func(runner_type, error_function, start)

    def arr_from_opts(opts: Options):
        return np.array([getattr(opts, k) for k, _ in sorted(fields.items())])

    # bds = (
    #     list((f.metadata.get("bounds")[0] for _, f in sorted(fields.items()))),
    #     list((f.metadata.get("bounds")[1] for _, f in sorted(fields.items())))
    # )
    bds = list((f.metadata.get("bounds") for _, f in sorted(fields.items())))
    print(bds)
    m = brute(f, bds, Ns=5)
    print(m)

    return construct_options(m)

    # r = GradientDescendRunner(Oracle(f, None), Vector(*arr_from_opts(start)), OldOptions(
    #     exit_condition=lambda st1, st2: st2.z != 10 ** 9 and ExitCondition.DELTA(tol)(st1, st2),
    #     a=Coef.GEOMETRIC_PROGRESSION(0.1, 0.9999)
    # ))
    # res = r.run()[0].steps[-1]
    # print(res)
    # return construct_options(res.point.coords), res.z


def _fields(cls):
    res = {}
    for filed in dataclasses.fields(cls):
        if filed.metadata.get("fixed"):
            continue
        res[filed.name] = filed

    return res


def _attempt(runner: tp.Type[AbstractRunner], func, opts: Options, attempts=1):
    res = []
    for t in range(attempts):
        # start = Vector(*[random.randint(-1, 1)] * func.dim)
        start = Vector(*[1] * func.dim)
        oracle = Oracle(func, func.target())
        b = BenchmarkResult.process_runner(runner(oracle, start, opts))
        res.append(b)
    return res


if __name__ == '__main__':
    m = optimize_params(
        WolfeRunner,
        ErrorFunction.accuracy,
        tol=0.0001
    )
    print(m)
# (NewtonSearchOptions(exit_condition_threshold=0.001, learning_rate=1.0000020137389727, grad_delta=0.0009998425454368135, search_max=10.000007937496651, search_iterations=10), 1.586603032594329)
# (NewtonSearchOptions(exit_condition_threshold=0.001, learning_rate=1.000000002888933, grad_delta=0.000999999997111067, search_max=10.000000014444666, search_iterations=10), 1.5388257750866914)
# (NewtonSearchOptions(exit_condition_threshold=0.001, learning_rate=1.0000004748111873, grad_delta=0.000999999529945541, search_max=9.999997625874334, search_iterations=9), 1.545624062564911)
# (NewtonSearchOptions(exit_condition_threshold=0.001, learning_rate=0.9999994154651817, grad_delta=0.0009999994154651817, search_max=9.999997077325908, search_iterations=9), 1.3626499050734104)
