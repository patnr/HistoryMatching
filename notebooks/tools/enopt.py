"""Importable EnOpt code (similar to 'optimize' notebook)."""

from dataclasses import dataclass

import numpy as np

from tools import utils
from tools.utils import center, apply, progbar


@dataclass
class nabla_ens:
    """Ensemble gradient estimate (LLS regression)."""
    chol:    float = 1.0   # Cholesky factor (or scalar std. dev.)
    nEns:    int   = 10    # Size of control perturbation ensemble
    precond: bool  = False # Use preconditioned form?
    # Will be used later:
    robustly:None  = None  # Method of treating robust objectives
    obj_ux:  None  = None  # Conditional objective function
    X:       None  = None  # Uncertainty ensemble

    def eval(self, obj, u, pbar):
        """Estimate `∇ obj(u)`"""
        U = utils.gaussian_noise(self.nEns, len(u), self.chol)
        dU = center(U)[0]
        dJ = self.obj_increments(obj, u, u + dU, pbar)
        if self.precond:
            g = dU.T @ dJ / (self.nEns-1)
        else:
            g = utils.rinv(dU, reg=.1, tikh=True) @ dJ
        return g

    def obj_increments(self, obj, u, U, pbar):
        return apply(obj, U, pbar=pbar)  # don't need to `center`


@dataclass
class backtracker:
    """Bisect until sufficient improvement."""
    sign:   int   = +1                                  # Search for max(+1) or min(-1)
    xSteps: tuple = tuple(.5**(i+1) for i in range(8))  # Trial step lengths
    rtol:   float = 1e-8                                # Convergence criterion
    nCPU:   int   = None

    def eval(self, obj, u0, J0, search_direction, pbar):
        atol = max(1e-8, abs(J0)) * self.rtol
        pbar.reset(len(self.xSteps))

        def step(xStep):
            du = self.sign * xStep * search_direction
            u1 = u0 + du
            J1 = obj(u1)
            dJ = J1 - J0
            return u1, J1, dJ

        for xSteps in split(self.xSteps, self.nCPU):
            results = apply(step, xSteps, pbar=False)
            for (u1, J1, dJ) in results:
                pbar.update()
                if self.sign*dJ > atol:
                    return u1, J1, dict(nDeclined=pbar.n)


def split(arr, step):
    """Split `arr` into segments of length `step`."""
    # NB: Other solutions (e.g. np.array_split) get surprisingly messy.
    # NB: The fallback `step` (length of individual batches) is cpu_count().
    #     The aim is to unblock as frequently as possible while utilizing all CPUs.
    if not step:
        import multiprocessing
        step = max(1, multiprocessing.cpu_count() - 1)
    return [arr[i:i+step] for i in range(0, len(arr), step)]


def GD(objective, u, nabla=nabla_ens(), line_search=backtracker(), nrmlz=True, nIter=100, quiet=False):
    """Gradient (i.e. steepest) descent/ascent."""

    # Reusable progress bars (limits flickering scroll in Jupyter) w/ short np printout
    with (progbar(total=nIter, desc="⏳ GD running", leave=True,  disable=quiet) as pbar_gd,
          progbar(total=10000, desc="→ grad. comp.", leave=False, disable=quiet) as pbar_en,
          progbar(total=10000, desc="→ line_search", leave=False, disable=quiet) as pbar_ls,
          np.printoptions(precision=2, threshold=2, edgeitems=1)):

        states = [[u, objective(u), {}]]

        for itr in range(nIter):
            u, J, info = states[-1]
            pbar_gd.set_postfix(u=f"{u}", obj=f"{J:.3g}📈")

            grad = nabla.eval(objective, u, pbar_en)
            info['grad'] = grad
            if nrmlz:
                grad /= np.sqrt(np.mean(grad**2))

            updated = line_search.eval(objective, u, J, grad, pbar_ls)
            pbar_gd.update()
            if updated:
                states.append(updated)
            else:
                info['cause'] = "✅ GD converged"
                break
        else:
            info['cause'] = "❌ GD ran out of iters"
        pbar_gd.set_description(info['cause'])

    return (np.asarray(arr) for arr in zip(*states))  # "transpose"
