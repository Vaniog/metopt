from utils import *
from coordinatedescendrunner import *


def main():
    def f(x: float, y: float) -> float:
        # return x ** 3 * y ** 5 * (4 - x - 7 * y)
        # return scipy.optimize.rosen((x, y))
        return (x - 2) ** 2 + (y + 1) ** 2 + x * y

    TARGET = (4 / 3, 20 / 63)
    PROBLEM = Problem(f, TARGET)
    print(f(*TARGET))
    runner = GradientDescendRunner(PROBLEM, (2, 1), Coef.CONST(0.0001),
                                   ExitCondition.NORM(Metric.EUCLID, 0.00001))
    runner.experiment(False, 25, plt_cfg=PlotConfig(-3, 3, func_num=100, dpi=500))


if __name__ == '__main__':
    main()
