import dataclasses
from collections import defaultdict
from dataclasses import dataclass
import typing as tp

from tabulate import tabulate
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from timeout_decorator import timeout_decorator
from typing_extensions import overload
from lab2.runners import *

from common.utils import *

MAX_INT = 10 ** 9


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
    @overload
    def compare(cls, runners: tp.Iterable[tp.Type[AbstractRunner]], params: tp.Dict):
        res = []
        for r in runners:
            print(f"--running {r.__name__}")
            res.append(cls.process_runner(r(**params)))
        return BenchmarkResult(res)

    @classmethod
    @overload
    def compare(cls, runners: tp.Iterable[AbstractRunner]):
        res = []
        for r in runners:
            print(f"--running {r.__class__.__name__}")
            res.append(cls.process_runner(r))
        return BenchmarkResult(res)

    @classmethod
    @overtake(runtime_type_checker='beartype')
    def compare(cls, *args, **kwargs):
        ...

    def print_results(self, sort_by: tp.Callable = None, name="RESULTS"):
        print(name)
        res = sorted(self.results, key=sort_by) if sort_by is not None else self.results
        print(tabulate(res, headers=self.results[0].headers(), tablefmt='orgtbl',
                       floatfmt=".8f"))
        print()

    def __repr__(self):
        return str(self.results)

    @classmethod
    def process_runner(cls, runner: AbstractRunner):
        try:
            res = cls._run(runner)
        except Exception as e:
            res = cls.ExperimentResult(
                runner.__class__.__name__,
                MAX_INT,
                MAX_INT,
                MAX_INT
            )
        return res

    def top(self, *fields: str, silent: bool = False, total=False):
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
    @overload
    def series(cls, runners: tp.Iterable[tp.Type[AbstractRunner]], params_it: tp.Iterable[dict]) \
            -> tp.List['BenchmarkResult']:
        res = []
        for i, p in enumerate(params_it):
            print(f"running experiment {i + 1}")
            res.append(cls.compare(runners, p))
        return res

    @classmethod
    @overload
    def series(cls, runners: tp.Iterable[tp.Iterable[AbstractRunner]]) \
            -> tp.List['BenchmarkResult']:
        res = []
        for i, rs in enumerate(runners):
            print(f"running experiment {i + 1}")
            res.append(cls.compare(rs))
        return res

    @classmethod
    @overtake(runtime_type_checker='beartype')
    def series(cls, *args, **kwargs):
        ...

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


if __name__ == '__main__':
    def f(x: float, y: float) -> float:
        return (x - 1) ** 2 + y ** 2


    TARGET = Vector(1, 0)
    PROBLEM = Oracle(f, TARGET)
    runners = list(map(lambda r: r(PROBLEM, Vector(3, 3)), RunnerMeta.runners))

    b = BenchmarkResult.compare(runners)
