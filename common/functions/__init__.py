import inspect

from .custom import *

_locals = locals()


def custom(dim: int = 2) -> tuple:
    fs = map(lambda x: x(dim), filter(lambda x: isinstance(x, type) and not inspect.isabstract(x), _locals.values()))
    filtered = filter(lambda f: f.dim == dim, fs)
    return tuple(filtered)
