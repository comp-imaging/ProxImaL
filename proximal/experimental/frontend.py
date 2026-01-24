import re

import numpy as np
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

from proximal.experimental.ir.lin_ops import Crop, FFTConv, Grad, MultiplyAdd
from proximal.experimental.ir.problem import Problem, Variable
from proximal.experimental.ir.prox_fns import GroupNorm, Nonneg, ProxFnBase, SumSquares
from proximal.experimental.models import LinOp


def getProxFn(name: str) -> type[ProxFnBase]:
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
        case "conv":
            return FFTConv(kernel=np.ones(3), input_dims=None, output_dims=None)
        case "grad":
            return Grad(input_dims=None, output_dims=None)
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
        scale_op, lin_op_option, crop_option, offset_option = visited_children

        # Decode scale value with default value 1.0
        scale: float = 1.0
        if isinstance(scale_op, list):
            scale = scale_op[0][0]

        # If crop operation exists, decode it.
        crop_op: Crop | None = crop_option[0] if isinstance(crop_option, list) else None

        # Decode offset value with default value 0.0
        offset: float | np.ndarray = 0.0
        if isinstance(offset_option, list):
            assert isinstance(offset_option[0][1], (float, np.ndarray))
            offset = offset_option[0][1]

        has_scaleoffset: bool = scale != 1.0 or isinstance(offset, np.ndarray) or offset != 0.0
        scaleoffset_op = MultiplyAdd(scale=scale, offset=offset)

        assert isinstance(lin_op_option, list)
        lin_op = lin_op_option[0]

        lin_ops: list[LinOp] = [] if isinstance(lin_op, Variable) else lin_op

        if crop_op is not None:
            lin_ops.append(crop_op)
        if has_scaleoffset:
            lin_ops.append(scaleoffset_op)

        return lin_ops

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

    def visit_CropOp(self, _, visited_children) -> LinOp:
        _, x_range, _, y_range, _ = visited_children

        return Crop(
            left=x_range.start,
            width=x_range.stop - x_range.start,
            top=y_range.start,
            height=y_range.stop - y_range.start,
        )

    def visit_LinOp(self, _, visited_children) -> list[LinOp]:
        name, _, expression, _ = visited_children

        lin_op = getLinOp(name)
        if isinstance(expression, Variable):
            return [lin_op]

        assert isinstance(expression, list)
        expression.append(lin_op)

        return expression

    def visit_Slice(self, _, visited_children) -> slice:
        start, _, stop = visited_children
        if start > stop:
            raise ValueError(f"Crop operator: expected start <= stop, found {start:d} < {stop:d}")
        return slice(start, stop)

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

    def visit_INTEGER(self, node, _) -> int:
        return int(node.text)

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
    ScaleOffset     = (NUMBER FACTOR_OPERATOR)* (ConvOp / LinOp / Variable) CropOp? (TERM_OPERATOR Offset)?
    ConvOp          = "conv(" ConstBuffer Comma ScaleOffset RPar
    CropOp          = "[" Slice Comma Slice "]"
    LinOp           = LinOpKey LPar ScaleOffset RPar
    Offset          = NUMBER / ConstBuffer

    Slice           = INTEGER ":" INTEGER

    TERM_OPERATOR   = ~"[-+]"
    FACTOR_OPERATOR = "*"
    NUMBER          = SCIENTIFIC / DECIMAL
    DECIMAL         = ~r"\d+\.?\d*"
    SCIENTIFIC      = ~r"\d+\.\d+e-?\d+"
    INTEGER         = ~r"[1-9]\d*" / "0"

    ProxFnKey       = "sum_squares" / "norm1" / "group_norm" / "nonneg"
    LinOpKey        = "grad"
    LPar            = "("
    RPar            = ")"
    Comma           = ","
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
