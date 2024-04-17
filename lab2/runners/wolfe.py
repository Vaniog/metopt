import dataclasses

import numpy as np

from common.utils import Vector
from lab2.runners.grad import grad
from lab2.runners.newton_const import NewtonConstOptions, NewtonConstRunner


@dataclasses.dataclass
class WolfeOptions(NewtonConstOptions):
    eps_armijo: float = 0.5
    eps_curvature: float = 0.75

    a0: float = 1
    teta_armija: float = 0.9

    search_iterations: int = 10

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
            if steps >= opts.search_iterations:
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
