from scipy.optimize import minimize
import typing as tp
from dataclasses import dataclass
from timeit import default_timer as timer

Vector2D = tp.Tuple[float, float]

@dataclass
class NudelMidRunner:
    functionType = tp.Callable[[Vector2D,Vector2D],float]
    func:functionType
    x0: Vector2D
    tol: float

    def run(selt):
        st = timer()
        res = minimize(selt.func, selt.x0, method='Nelder-Mead',tol = selt.tol)
        end = timer()
        print(res)

def main():
    def f(point):
        x,y = point
        return x ** 3 * y ** 5 * (4 - x - 7 * y)
    
    runner = NudelMidRunner(f,[10,10],1e-2)
    runner.run()

if __name__ == '__main__':
    main()

