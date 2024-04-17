import dataclasses

import numpy as np
from numpy.linalg import norm

from common.utils import Vector
from lab2.runners.grad import grad
from lab2.runners.newton_const import NewtonConstOptions, NewtonConstRunner


@dataclasses.dataclass
class WolfeOptions(NewtonConstOptions):
    eps_armijo: float = dataclasses.field(default=0.5, metadata={"fixed": True})
    eps_curvature: float = dataclasses.field(default=0.75, metadata={"fixed": True})

    a0: float = dataclasses.field(default=1, metadata={"fixed": True})
    teta_armija: float = dataclasses.field(default=0.9, metadata={"fixed": True})

    search_iterations: float = dataclasses.field(default=0.5, metadata={"bounds": (0, 0.001)})
    # acc: float = dataclasses.field(default=3, metadata={"bounds": (0, 5)})

    def validate(self):
        return all((
            super().validate(),
            #     я помню что тут кто-то должен быть меньше кого-то,
            #     но кто есть кто, я не помню
        ))


class WolfeRunner(NewtonConstRunner):
    opts: WolfeOptions

    def wolfe_search(self, f, x: np.ndarray, s: np.ndarray, opts: WolfeOptions) -> float:
        fx = f(*x)
        dfsk = np.dot(grad(f, x, opts.grad_delta), s)

        def armijo(a: float) -> bool:
            return f(*(x + s * a)) - fx <= opts.eps_armijo * a * dfsk

        def curvature(a: float) -> bool:
            return np.dot(grad(f, x + s * a, opts.grad_delta), s) >= opts.eps_curvature * dfsk

        a = opts.a0
        teta_curvature = (1 + 1 / opts.teta_armija) / 2

        steps = 0
        while self.running():
            steps += 1
            if steps >= opts.search_iterations * 30000:
                return a
            if not armijo(a):
                a *= opts.teta_armija
                continue
            if not curvature(a):
                a *= teta_curvature
                continue
            break

        return a

    def _run(self, start: Vector, *args, **kwargs):
        prev = np.array(start.coords)
        cur = np.array(start.coords)
        while self.running():
            sk = self.sk(cur)

            cur = cur + self.sk(cur) * self.wolfe_search(self.o.f, cur, sk, self.opts)

            if self.opts.exit_condition(prev, cur):
                break
            prev = cur
