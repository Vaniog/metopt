from common.benchmark import BenchmarkResult
from common.functions import Basic
from common.utils import Oracle, Vector
from lab2.runners import WolfeRunner


def main():
    f = Basic(2)
    TARGET = f.target()
    PROBLEM = lambda: Oracle(f, TARGET)
    runners = list(map(lambda r: r(PROBLEM(), Vector(3, 2)), [WolfeRunner]))

    b = BenchmarkResult.compare(runners)
    b.top("accuracy")
    pass


if __name__ == '__main__':
    main2()
