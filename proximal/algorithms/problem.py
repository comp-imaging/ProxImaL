from . import admm
from . import pock_chambolle as pc
from . import half_quadratic_splitting as hqs
from . import linearized_admm as ladmm
from proximal.utils.utils import Impl
from proximal.lin_ops import Variable, CompGraph, est_CompGraph_norm, vstack
from proximal.prox_fns import ProxFn
from . import absorb
from . import merge
import numpy as np

NAME_TO_SOLVER = {
    "admm": admm,
    "pock_chambolle": pc,
    "pc": pc,
    "half_quadratic_splitting": hqs,
    "hqs": hqs,
    "linearized_admm": ladmm,
    "ladmm": ladmm,
}


class Problem(object):
    """An object representing a convex optimization problem.
    """

    def __init__(self, prox_fns,
                 implem=Impl['numpy'], try_diagonalize=True,
                 absorb=True, merge=True,
                 try_split=True, try_fast_norm=True, scale=True,
                 psi_fns=None, omega_fns=None,
                 lin_solver="cg", solver="pc"):
        # Accept single function as argument.
        if isinstance(prox_fns, ProxFn):
            prox_fns = [prox_fns]
        self.prox_fns = prox_fns
        self.implem = implem
        self.try_diagonalize = try_diagonalize  # Auto diagonalize?
        self.try_split = try_split  # Auto partition?
        self.try_fast_norm = try_fast_norm  # Fast upper bound on ||K||?
        self.scale = scale  # Auto scale problem?
        self.absorb = absorb  # Absorb lin ops into prox fns?
        self.merge = merge  # Merge prox fns?

        # Defaults for psi and omega fns.
        # Should have psi_fns + omega_fns == prox_fns
        if psi_fns is None and omega_fns is None:
            psi_fns = []
            omega_fns = []
        elif psi_fns is None:
            psi_fns = [fn for fn in prox_fns if fn not in omega_fns]
        elif omega_fns is None:
            omega_fns = [fn for fn in prox_fns if fn not in psi_fns]
        else:
            assert set(psi_fns + omega_fns) == set(prox_fns)
        self.omega_fns = omega_fns
        self.psi_fns = psi_fns

        self.solver = solver
        self.lin_solver = lin_solver

    def set_absorb(self, absorb):
        """Try to absorb lin ops in prox fns?
        """
        self.absorb = absorb

    def set_merge(self, merge):
        """Try to merge prox fns?
        """
        self.merge = merge

    def set_automatic_frequency_split(self, freq_split):
        self.freq_split = freq_split

    def set_implementation(self, implem=Impl['numpy']):
        """Set the implementation of the lin ops and proxes.
        """
        self.implem = implem

    def set_solver(self, solver):
        """Set the solver.
        """
        self.solver = solver

    def set_lin_solver(self, lin_solver):
        """Set solver for linear systems/least squares.
        """
        self.lin_solver = lin_solver

    def solve(self, solver=None, *args, **kwargs):
        if solver is None:
            solver = self.solver

        if len(self.omega_fns + self.psi_fns) == 0:
            prox_fns = self.prox_fns
        else:
            prox_fns = self.omega_fns + self.psi_fns
        # Absorb lin ops if desired.
        if self.absorb:
            prox_fns = absorb.absorb_all_lin_ops(prox_fns)

        # Merge prox fns.
        if self.merge:
            prox_fns = merge.merge_all(prox_fns)
        # Absorb offsets.
        prox_fns = [absorb.absorb_offset(fn) for fn in prox_fns]
        # TODO more analysis of what solver to use.
        # Short circuit with one function.
        if len(prox_fns) == 1 and type(prox_fns[0].lin_op) == Variable:
            fn = prox_fns[0]
            var = fn.lin_op
            var.value = fn.prox(0, np.zeros(fn.lin_op.shape))
            return fn.value
        elif solver in NAME_TO_SOLVER:
            module = NAME_TO_SOLVER[solver]
            if len(self.omega_fns + self.psi_fns) == 0:
                if self.try_split and len(prox_fns) > 1 and len(self.variables()) == 1:
                    psi_fns, omega_fns = module.partition(prox_fns,
                                                          self.try_diagonalize)
                else:
                    psi_fns = prox_fns
                    omega_fns = []
            # Scale the problem.
            if self.scale:
                K = CompGraph(vstack([fn.lin_op for fn in psi_fns]),
                              implem=self.implem)
                Knorm = est_CompGraph_norm(K, try_fast_norm=self.try_fast_norm)
                for idx, fn in enumerate(psi_fns):
                    psi_fns[idx] = fn.copy(fn.lin_op / Knorm,
                                           beta=fn.beta * np.sqrt(Knorm),
                                           implem=self.implem)
                for idx, fn in enumerate(omega_fns):
                    omega_fns[idx] = fn.copy(beta=fn.beta / np.sqrt(Knorm),
                                             implem=self.implem)
            opt_val = module.solve(psi_fns, omega_fns,
                                   lin_solver=self.lin_solver,
                                   try_diagonalize=self.try_diagonalize,
                                   try_fast_norm=self.try_fast_norm,
                                   scaled=self.scale,
                                   *args, **kwargs)
            # Unscale the variables.
            if self.scale:
                for var in self.variables():
                    var.value /= np.sqrt(Knorm)
            return opt_val
        else:
            raise Exception("Unknown solver.")

    def variables(self):
        """Return a list of variables in the problem.
        """
        vars_ = []
        for fn in self.prox_fns:
            vars_ += fn.variables()
        return list(set(vars_))
