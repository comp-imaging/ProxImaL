from dataclasses import dataclass

from jinja2 import Environment, PackageLoader
from numpy import ndarray

from proximal.experimental.ir.lin_ops import FFTConv, Grad
from proximal.experimental.ir.problem import Problem
from proximal.experimental.ir.prox_fns import FFTConvSumSquares, GroupNorm, Nonneg, SumSquares, WeightedLeastSquares
from proximal.experimental.models import LinOp, ProxFn

env = Environment(loader=PackageLoader("proximal"))


def getProxFnImpl(prox_fn: ProxFn) -> str:
    match prox_fn:
        case FFTConvSumSquares():
            return "least_square_direct"

        case WeightedLeastSquares():
            return """[](const Func& v, const Expr& tau) {
    using Halide::_;

    Func proxWeightedL2{"proxWeightedL2"};
    const Expr w = weight(_);
    proxWeightedL2(_) = (w * b * 2 * tau + v(_)) / (w * w * 2 * tau + 1);
    return proxWeightedL2;
}"""

        case SumSquares():
            # It is far more direct to implement proxL2Norm here.
            return """[](const Func& v, const Expr& tau) {
    using Halide::_;
    Func proxL2{"proxL2"};
    proxL2(_) = v(_) * (1 / tau) / ((1 / tau)+ 2);
    return proxL2;
}"""

        case GroupNorm():
            return "proxIsoL1"

        case Nonneg():
            return """[](const Func& v, const Expr& rho) {
    using Halide::_;
    Func nonneg{"nonneg"};
    nonneg(_) = max(v(_), 0.0f);
    return nonneg;
}"""

    raise RuntimeError(f"Proximal function {type(prox_fn).__name__} not found")


@dataclass
class HalideLinOpImpl:
    forward_func: str
    adjoint_func: str
    stage_name: str


def getLinOpImpl(lin_op: LinOp) -> HalideLinOpImpl:
    match lin_op:
        case Grad():
            return HalideLinOpImpl("K_grad_mat", "KT_grad_mat", "gradient")

        case FFTConv():
            return HalideLinOpImpl("convImg", "convImgT", "blurred")

    raise RuntimeError(f"Linear operator {type(lin_op).__name__} not found")


@dataclass
class TransformCode:
    stages: list[str]
    output_name: str


def generateForwardImpl(lin_ops: list[LinOp], input_name: str = "u") -> TransformCode:
    if len(lin_ops) == 0:
        # Passthrough
        return TransformCode([], input_name)

    halide_code: list[str] = []
    for lin_op in lin_ops:
        impl = getLinOpImpl(lin_op)
        output_name = impl.stage_name

        halide_code.append(f"""
const Func {output_name} = {impl.forward_func}({input_name}, width, height);
""")
        input_name = output_name

    return TransformCode(halide_code, output_name)


def generateAdjointImpl(lin_ops: list[LinOp], input_name: str = "z") -> TransformCode:
    if len(lin_ops) == 0:
        # Passthrough
        return TransformCode([], input_name)

    halide_code: list[str] = []
    for lin_op in reversed(lin_ops):
        impl = getLinOpImpl(lin_op)
        output_name = impl.stage_name

        halide_code.append(f"""
const Func {output_name} = {impl.adjoint_func}({input_name}, width, height);
""")
        input_name = output_name

    return TransformCode(halide_code, output_name)


class LinearizedADMM:
    def __init__(self, problem: Problem, mu: float = 1.0):
        self.mu = mu
        self.rho = problem.Knorm / mu
        self.omega_fn = problem.omega_fn
        self.psi_fns = problem.psi_fns
        self.u = problem.u

    def generateConfig(self) -> str:
        """Generate "problem_config.h" representing the input-output image
        dimensions."""
        assert self.omega_fn is not None
        assert hasattr(self.omega_fn, "b")

        b = self.omega_fn.new_b if hasattr(self.omega_fn, "new_b") else self.omega_fn.b

        assert isinstance(b, ndarray), f"Measured image is not a Numpy array"

        config_template = env.get_template("problem-config.h.j2")
        return config_template.render(
            psi_fns=self.psi_fns,
            omega_fn=self.omega_fn,
            u=self.u,
            b=b,
        )

    def generateProblemDefinition(self) -> str:
        # halide_code.append(f"return {{ {', '.join(output_names)} }};")
        definition_template = env.get_template("problem-definition.h.j2")
        return definition_template.render(
            psi_fns=zip(self.psi_fns, [getProxFnImpl(fn) for fn in self.psi_fns]),
            omega_fn=self.omega_fn,
            omega_fn_name=type(self.omega_fn).__name__,
            forward_op=[generateForwardImpl(fn.lin_ops) for fn in self.psi_fns],
            adjoint_op=[generateAdjointImpl(fn.lin_ops, f"z[{i}]") for i, fn in enumerate(self.psi_fns)],
            u=self.u,
        )
