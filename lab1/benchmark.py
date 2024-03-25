import dataclasses
from collections import defaultdict
from dataclasses import dataclass
import typing as tp

from lab1.runners import AbstractRunner, CoordinateGradientRunner, GradientDescendRunner, Vector, Oracle, Coef, \
    ExitCondition, Metric, RunnerMeta
from tabulate import tabulate
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


class BenchmarkResult:
    results: tp.List['ExperimentResult']

    # noinspection PyTypeChecker
    @dataclass
    class ExperimentResult:
        name: str
        accuracy: float | None
        queries: int
        time: float

        def __iter__(self):
            return iter(dataclasses.asdict(self).values())

        def headers(self):
            return list(dataclasses.asdict(self).keys())

        def parameters(self):
            return list(filter(lambda el: el != 'name', dataclasses.asdict(self).keys()))

    def __init__(self, results: tp.List['ExperimentResult']):
        self.results = results

    @classmethod
    def compare(cls, runners: tp.Iterable[tp.Type[AbstractRunner]], params: tp.Dict):
        res = []
        for r in runners:
            print(f"--running {r.__name__}")
            res.append(cls.process_runner(r, params))
        return BenchmarkResult(res)

    def print_results(self, sort_by: tp.Callable = None, name="RESULTS"):
        print(name)
        res = sorted(self.results, key=sort_by) if sort_by is not None else self.results
        print(tabulate(res, headers=self.results[0].headers(), tablefmt='orgtbl',
                       floatfmt=".8f"))
        print()

    def __repr__(self):
        return str(self.results)

    @classmethod
    def process_runner(cls, runner_tp: tp.Type[AbstractRunner], params: tp.Dict):
        # noinspection PyArgumentList
        runner = runner_tp(**params)

        res = cls._run(runner)
        return res

    def top(self, *fields: str, silent: bool = False, total=True):
        score = defaultdict(int)
        for f in fields:
            sort_by = lambda r: getattr(r, f)
            for i, res in enumerate(sorted(self.results, key=sort_by)):
                score[res.name] += i + 1
            if not silent:
                self.print_results(sort_by=sort_by, name=f"sorted by {f}")

        if not silent and total:
            print()
            print("total results (less score means more efficient)")
            fmt = []
            for it in sorted(score.items(), key=lambda el: el[1]):
                fmt.append(it)
            print(tabulate(fmt, headers=("name", "score"), tablefmt='orgtbl'))

        if silent:
            return score

    @classmethod
    def _run(cls, runner: AbstractRunner) -> ExperimentResult:
        res, time = runner.run()
        acc = res.accuracy(runner.o.target)

        return cls.ExperimentResult(
            runner.__class__.__name__,
            acc,
            res.queries(),
            time,
        )

    @classmethod
    def series(cls, runners: tp.Iterable[tp.Type[AbstractRunner]], params_it: tp.Iterable[dict]) \
            -> tp.List['BenchmarkResult']:
        res = []
        for i, p in enumerate(params_it):
            print(f"running experiment {i + 1}")
            res.append(cls.compare(runners, p))
        return res

    @classmethod
    def variate(cls, runners: tp.Iterable[tp.Type[AbstractRunner]], params: tp.Dict, param_name: str,
                rng: tp.Iterable[tp.Any]) -> tp.List['BenchmarkResult']:
        def gen():
            for p_value in rng:
                p = dict(**params)
                p[param_name] = p_value
                yield p

        return cls.series(runners, gen())

    @classmethod
    def plot_results(cls, results: tp.List['BenchmarkResult'], fields: tp.Tuple, names: tp.List[str]):
        assert len(results) > 0
        assert len(fields) < 10
        runner_names = list(map(lambda el: el.name, results[0].results))
        colors = list(mcolors.TABLEAU_COLORS.values())
        assert len(runner_names) <= len(colors)
        plt.figure(figsize=(8, 4 * len(fields)), dpi=300)
        for i, f in enumerate(fields):
            fig = plt.subplot(int(f"{len(fields)}1{i + 1}"))
            fig.set_title(f)

            for j, runner in enumerate(runner_names):
                values = []

                for r in results:
                    if f == 'top':
                        values.append(r.top(*r.results[0].parameters(), silent=True)[runner])
                    else:
                        values.append(getattr(r.results[j], f))

                plt.plot(names, values, color=colors[j], label=runner)
            if i == 0:
                plt.legend()

        # plt.tight_layout()
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45),
        #            fancybox=True, shadow=True, ncol=5)
        plt.show()


def main():
    def f(x: float, y: float) -> float:
        # return x ** 3 * y ** 5 * (4 - x - 7 * y)
        # return scipy.optimize.rosen((x, y))
        return (x - 1) ** 2 + y ** 2
        # return x ** 2 + y ** 2

        # TARGET = Vector(4 / 3, 20 / 63)

    TARGET = Vector(1, 0)
    PROBLEM = Oracle(f, TARGET)
    b = BenchmarkResult.compare(RunnerMeta.runners, dict(
        o=PROBLEM,
        start=Vector(2, 1),
        a=Coef.CONST(0.0001),
        exit_condition=ExitCondition.NORM(Metric.EUCLID, 0.00001)
    ))
    # b.print_results(sort_by=lambda el: el.queries, name="sorted by queries")
    # b.print_results(sort_by=lambda el: el.accuracy, name="sorted by accuracy")
    # b.print_results(sort_by=lambda el: el.time, name="sorted by time")

    # b.top("accuracy", "time", "queries")


def main2():
    def f(x: float, y: float) -> float:
        # return x ** 3 * y ** 5 * (4 - x - 7 * y)
        # return scipy.optimize.rosen((x, y))
        return (x - 1) ** 2 + y ** 2
        # return x ** 2 + y ** 2

        # TARGET = Vector(4 / 3, 20 / 63)

    TARGET = Vector(1, 0)
    PROBLEM = Oracle(f, TARGET)
    # (GradientDescendRunner, CoordinateGradientRunner)
    bs = BenchmarkResult.variate(RunnerMeta.runners, dict(
        o=PROBLEM,
        a=Coef.CONST(0.0001),
        exit_condition=ExitCondition.NORM(Metric.EUCLID, 0.00001)
    ), "start", (Vector(2, 1), Vector(2, 2), Vector(2, 3),))

    BenchmarkResult.plot_results(bs, ("queries", "accuracy", "time", "top"), names=["1", "2", "3"])


if __name__ == '__main__':
    main2()
