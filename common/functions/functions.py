from OptimizationTestFunctions import plot_3d

from common.functions import custom
from common.utils import Vector


class __TupleWithSize(tuple):

    @property
    def size(self):
        return len(self)


def _decorate(f):
    inner_call = f.__call__

    def my_call(self, *args):
        if len(args) == 0:
            raise Exception("???")
        if len(args) == 1:
            return inner_call(args[0])
        else:
            return inner_call(__TupleWithSize(args))

    type(f).__call__ = my_call
    f.__name__ = f.__class__.__name__
    f.target = lambda self: Vector(*self.x_best)
    return f


def _from_lib(dim: int) -> tuple:
    from OptimizationTestFunctions import Abs, Fletcher, Stairs, Rosenbrock
    res = []
    for f in filter(lambda x: isinstance(x, type), locals().values()):
        func = f(dim)
        res.append(_decorate(func))
    return tuple(res)


def functions(dim: int = 2):
    return _from_lib(dim) + custom(dim)


if __name__ == '__main__':
    for f in functions():
        plot_3d(f, title=f.__name__)
        print(f(2, 3))
