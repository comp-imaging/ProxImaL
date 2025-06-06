import re

import numpy as np
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

from proximal.experimental.ir.lin_ops import Crop, FFTConv, Grad, MultiplyAdd
from proximal.experimental.ir.problem import Problem, Variable
from proximal.experimental.ir.prox_fns import GroupNorm, Nonneg, ProxFnBase, SumSquares
from proximal.experimental.models import LinOp


def getProxFn(name: str) -> ProxFnBase:
    match name:
        case "sum_squares":
            return SumSquares

        case "group_norm":
            return GroupNorm

        case "nonneg":
            return Nonneg

        case _:
            raise RuntimeError("ProxFn {name} not found")


def getLinOp(name: str) -> LinOp:
    match name:
        case "grad":
            return Grad()
        case "crop":
            return Crop(left=0, top=0, width=10, height=10)
        case _:
            raise RuntimeError(f"Linear operator {name} not found")


class ProxImaLDSLVisitor(NodeVisitor):
    def __init__(self, const_buffers: dict):
        self.const_buffers = const_buffers

    def visit_Problem(self, _, visited_children) -> list[ProxFnBase]:
        first, others = visited_children

        prox_fns = [first]
        if not isinstance(others, list):
            return prox_fns

        for _, fn in others:
            prox_fns.append(fn)

        return prox_fns

    def visit_ScaledProxFn(self, _, visited_children) -> ProxFnBase:
        factor_op, prox_fn = visited_children

        if not isinstance(factor_op, list):
            return prox_fn

        for factor, _ in factor_op:
            prox_fn.alpha *= factor

        assert factor > 0.0
        return prox_fn

    def visit_ProxFn(self, _, visited_children) -> ProxFnBase:
        name, _, expression, _ = visited_children

        if isinstance(expression, Variable):
            return getProxFn(name)(lin_ops=[])

        return getProxFn(name)(lin_ops=expression)

    def visit_EXPRESSION(self, _, visited_children):
        return visited_children[0]

    def visit_ScaleOffset(self, _, visited_children) -> list[LinOp]:
        scale_op, lin_op, offset_op = visited_children

        scale = 1.0
        if isinstance(scale_op, list):
            scale = scale_op[0][0]

        offset: float | np.ndarray = 0.0
        if isinstance(offset_op, list):
            assert isinstance(offset_op[0][1], (float, np.ndarray))
            offset = offset_op[0][1]

        has_scaleoffset: bool = scale != 1.0 or isinstance(offset, np.ndarray) or offset != 0.0
        new_linop = MultiplyAdd(scale=scale, offset=offset)
        assert isinstance(lin_op, list)
        if isinstance(lin_op[0], Variable):
            return [new_linop] if has_scaleoffset else []

        assert isinstance(lin_op[0][0], LinOp)
        if has_scaleoffset:
            lin_op[0].append(new_linop)

        return lin_op[0]

    def visit_ConstBuffer(self, node, _) -> np.ndarray:
        name = node.text
        assert name in self.const_buffers
        return self.const_buffers[name]

    def visit_ConvOp(self, _, visited_children) -> list[LinOp]:
        _, kernel, _, expression, _ = visited_children

        assert isinstance(kernel, np.ndarray)
        lin_op = FFTConv(kernel=kernel)

        if isinstance(expression, Variable):
            return [lin_op]

        assert isinstance(expression, list)
        expression.append(lin_op)

        return expression

    def visit_LinOp(self, _, visited_children) -> list[LinOp]:
        name, _, expression, _ = visited_children

        lin_op = getLinOp(name)
        if isinstance(expression, Variable):
            return [lin_op]

        assert isinstance(expression, list)
        expression.append(lin_op)

        return expression

    def visit_Offset(self, _, visited_children) -> float:
        # TODO Support constant buffer
        return visited_children[0]

    def visit_TERM_OPERATOR(self, node, _) -> str:
        return node.text

    def visit_ProxFnKey(self, node, _) -> str:
        return node.text

    def visit_LinOpKey(self, node, _) -> str:
        return node.text

    def visit_Variable(self, node, _) -> Variable:
        return Variable([5, 5, 5])

    def visit_NUMBER(self, node, _) -> float:
        return float(node.text)

    def generic_visit(self, node, visited_children):
        """The generic visit method."""
        return visited_children or node


def parse(
    expression: str,
    variable: str = "u",
    variable_dims: list[int] = [5, 5, 5],
    const_buffers: dict = {},
) -> Problem:
    grammar = Grammar(
        r"""
    Problem         = ScaledProxFn ("+" ScaledProxFn)*
    ScaledProxFn    = (NUMBER FACTOR_OPERATOR)* ProxFn
    ProxFn          = ProxFnKey LPar ScaleOffset RPar
    ScaleOffset     = (NUMBER FACTOR_OPERATOR)* (ConvOp / LinOp / Variable) (TERM_OPERATOR Offset)?
    ConvOp          = "conv(" ConstBuffer "," ScaleOffset RPar
    LinOp           = LinOpKey LPar ScaleOffset RPar
    Offset          = NUMBER / ConstBuffer

    TERM_OPERATOR   = ~"[-+]"
    FACTOR_OPERATOR = "*"
    NUMBER          = SCIENTIFIC / DECIMAL
    DECIMAL         = ~r"\d+\.?\d*"
    SCIENTIFIC      = ~r"\d+\.\d+e-?\d+"

    ProxFnKey       = "sum_squares" / "norm1" / "group_norm" / "nonneg"
    LinOpKey        = "grad" / "crop"
    LPar            = "("
    RPar            = ")"
    """
        f"""ConstBuffer     = "{'"/"'.join([key for key in const_buffers])}"
"""
        f'Variable        = "{variable:s}"\n'
    )

    # Remove all newlines and whitespaces, and then parse the expression.
    # TODO: Build the whitespace logic into Parismonious.
    ast = grammar.parse(re.sub(r"[ \t\r\n]*", "", expression))

    visitor = ProxImaLDSLVisitor(const_buffers)
    prox_fns = visitor.visit(ast)
    return Problem(prox_fns, Variable(variable_dims))
