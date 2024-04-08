import numpy as np
from numpy.linalg import inv


def grad(f, p: np.ndarray, delta: float) -> np.ndarray:
    ds = []

    fp = f(*p)
    for i, _ in enumerate(p):
        coords = [*p]
        coords[i] += delta
        ds.append(f(*coords) - fp)

    res = np.array(list(map(lambda el: el / delta, ds)))
    return res


def grad2(f, p: np.ndarray, delta: float) -> np.ndarray:
    dgs = []

    gp = grad(f, p, delta)
    for i, _ in enumerate(p):
        coords = [*p]
        coords[i] += delta
        dgs.append(grad(f, np.array(coords), delta) - gp)

    res = np.array(
        list(map(lambda el: el * (1 / delta), dgs)))
    return res


def newton_dir(f, p: np.ndarray, delta: float) -> np.ndarray:
    grad2inv = inv(grad2(f, p, delta))
    a = grad2inv @ grad(f, p, delta)
    return -a
