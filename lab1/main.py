from runners import *


def main():
    def f(x: float, y: float) -> float:
        # return x ** 3 * y ** 5 * (4 - x - 7 * y)
        # return scipy.optimize.rosen((x, y))
        return (x - 2) ** 2 + (y + 1) ** 2 + x * y
        # return x ** 2 + y ** 2

    # TARGET = Vector(4 / 3, 20 / 63)
    TARGET = Vector(0, 0)
    PROBLEM = Oracle(f, TARGET)
    print(f(*TARGET))
    runner = GradientDescendRunner(PROBLEM, Vector(2, 1), Coef.CONST(0.0001),
                                   ExitCondition.NORM(Metric.EUCLID, 0.00001))
    runner.experiment(False, 25, plt_cfg=PlotConfig(-3, 3, func_lines_num=100, dpi=500, draw_function=True))


def main2():
    def f(x: float, y: float, z: float) -> float:
        # return x ** 3 * y ** 5 * (4 - x - 7 * y)
        # return scipy.optimize.rosen((x, y))
        return (x - 1) ** 2 + y ** 2 + z ** 2
        # return x ** 2 + y ** 2

    # TARGET = Vector(4 / 3, 20 / 63)
    TARGET = Vector(1, 0, 0)
    PROBLEM = Oracle(f, TARGET)
    print(f(*TARGET))
    runner = CoordinateGradientRunner(PROBLEM, Vector(2, 1, 4), Coef.CONST(0.0001),
                                      ExitCondition.NORM(Metric.EUCLID, 0.00001))
    runner.experiment(False, 25, plt_cfg=PlotConfig(-3, 3, func_lines_num=100, dpi=500, draw_function=True))


def main3():
    def f(x: float, y: float) -> float:
        # return x ** 3 * y ** 5 * (4 - x - 7 * y)
        # return scipy.optimize.rosen((x, y))
        return (x - 1) ** 2 + y ** 2
        # return x ** 2 + y ** 2

    # TARGET = Vector(4 / 3, 20 / 63)
    TARGET = Vector(1, 0)
    PROBLEM = Oracle(f, TARGET)
    print(f(*TARGET))
    runner = NelderMeadRunner(PROBLEM, Vector(2, 1), Coef.CONST(0.0001),
                                             ExitCondition.NORM(Metric.EUCLID, 0.00001))
    runner.experiment(False, 25, plt_cfg=PlotConfig(-3, 3, func_lines_num=100, dpi=500))


if __name__ == '__main__':
    main3()
