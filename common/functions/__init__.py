from custom import *

_locals = locals()


def custom(dim: int = 2) -> tuple:
    return tuple(map(lambda x: x(dim), filter(lambda x: isinstance(x, type) and x.__name__ != "Base", _locals.values())))
