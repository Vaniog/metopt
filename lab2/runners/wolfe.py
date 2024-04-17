import dataclasses

import numpy as np
from numpy.linalg import norm

from common.utils import Vector
from lab2.runners.grad import grad
from lab2.runners.newton_const import NewtonConstOptions, NewtonConstRunner


@dataclasses.dataclass
class WolfeOptions(NewtonConstOptions):
    eps_armijo: float = dataclasses.field(default=0.00018576569900265817, metadata={"bounds": (0, 1)})
    eps_curvature: float = dataclasses.field(default=0.8941550352184215, metadata={"bounds": (0, 1)})

    a0: float = dataclasses.field(default=0.2651735328477399, metadata={"bounds": (0, 1)})
    teta_armija: float = dataclasses.field(default=0.40125631563010494, metadata={"bounds": (0, 1)})

    search_iterations: float = dataclasses.field(default=0.9628174281615451, metadata={"bounds": (0, 1)})
    # acc: float = dataclasses.field(default=3, metadata={"bounds": (0, 5)})

    def validate(self):
        return all((
            super().validate(),
            #     я помню что тут кто-то должен быть меньше кого-то,
            #     но кто есть кто, я не помню
        ))
# WolfeOptions(exit_condition_threshold=0.001, learning_rate=1, grad_delta=0.001, eps_armijo=0.00018576569900265817, eps_curvature=0.8941550352184215, a0=0.2651735328477399, teta_armija=0.40125631563010494, search_iterations=0.9628174281615451)

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
            if steps >= opts.search_iterations * 20:
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
