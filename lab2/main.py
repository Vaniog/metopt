from common.benchmark import BenchmarkResult
from common.functions import Basic
from common.utils import Oracle, Vector, PlotConfig
from lab2.runners import WolfeRunner


def main():
    f = Basic(2)
    TARGET = f.target()
    PROBLEM = lambda: Oracle(f, TARGET)
    runners = list(map(lambda r: r(PROBLEM(), Vector(3, 2)), [WolfeRunner]))

    b = BenchmarkResult.compare(runners)
    b.top("accuracy")
    pass


def main2():
    # from lab1.runners import GradientDescendRunner
    # from common.functions import functions
    # f = functions(include=["Rosenbrock"])[0]
    # r = GradientDescendRunner(Oracle(f, f.target()), Vector(2, 2), None)
    # print(r.experiment(plt_cfg=PlotConfig(draw_steps=False)))

    from common.functions import functions
    print(functions(exclude=["Abs", "Stairs"]))


if __name__ == '__main__':
    main2()
