import inspect

from .custom_functions import *

_locals = locals()


def custom(dim: int = 2) -> tuple:
    fs = map(lambda x: x(dim), filter(lambda x: isinstance(x, type) and issubclass(x, Base) and not inspect.isabstract(x), _locals.values()))
    filtered = filter(lambda f: f.dim == dim, fs)
    return tuple(filtered)


class __TupleWithSize(tuple):
    @property
    def size(self):
        return len(self)


def _decorate(f, dim):
    inner_call = f.__call__

    def my_call(self, *args):
        if len(args) == 0:
            raise Exception("???")
        if len(args) == 1:
            return inner_call(args[0])
        else:
            return inner_call(__TupleWithSize(args))

    def target():
        return Vector(*f.x_best)

    type(f).__call__ = my_call
    f.__name__ = f.__class__.__name__
    f.target = target
    f.dim = dim
    return f


def _from_lib(dim: int) -> tuple:
    from OptimizationTestFunctions import Abs, Fletcher, Stairs, Rosenbrock
    res = []
    for f in filter(lambda x: isinstance(x, type), locals().values()):
        func = f(dim)
        res.append(_decorate(func, dim))
    return tuple(res)


def functions(dim: int = 2, exclude: list = None, include: list = None):
    if not exclude:
        exclude = []
    t = (lambda f: f.__name__ in include) if include else lambda f: True
    return list(filter(lambda f: f.__name__ not in exclude and t(f), _from_lib(dim) + custom(dim)))


if __name__ == '__main__':
    from OptimizationTestFunctions import plot_3d

    for f in functions():
        plot_3d(f, title=f.__name__)
