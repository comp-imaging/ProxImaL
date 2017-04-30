from .lin_op import LinOp
from ..utils.cuda_codegen import indent, sub2ind, ind2subCode, NodeReverseInOut, ReverseInOut
import numpy as np


class vstack(LinOp):
    """Vectorizes and stacks inputs.
    """

    def __init__(self, input_nodes, implem=None):
        height = sum([node.size for node in input_nodes])
        self.input_shapes = list([i.shape for i in input_nodes])
        super(vstack, self).__init__(input_nodes, (height,))

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        offset = 0
        for idx, input_data in enumerate(inputs):
            size = input_data.size
            outputs[0][offset:size + offset] = input_data.flatten()
            offset += size

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        offset = 0
        for idx, output_data in enumerate(outputs):
            size = output_data.size
            data = inputs[0][offset:size + offset]
            output_data[:] = np.reshape(data, output_data.shape)
            offset += size

    def forward_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        #print("vstack:forward:cuda")
        # multiple reshaped output in, linear index out
        res = "var_%(num_tmp_vars)d" % locals()
        code = """/*vstack*/
float %(res)s = 0;
""" % locals()
        num_tmp_vars += 1
        offset = 0
        input_nodes = cg.input_nodes(self)
        assert (len(abs_idx) == 1 and len(input_nodes) == len(self.input_shapes))
        abs_idx = abs_idx[0]
        for idx, node in enumerate(input_nodes):
            endoffset = offset + node.size
            sub_idx_vars = list(["idx_%d" % d for d in range(num_tmp_vars, num_tmp_vars + len(self.input_shapes[idx]))])
            num_tmp_vars += len(self.input_shapes[idx])

            sub_idx_var_defs = indent(ind2subCode("%s - %d" % (abs_idx, offset), self.input_shapes[idx], sub_idx_vars), 4)
            sub_idx_var_defs = indent(sub_idx_var_defs,4)

            icode, var, num_tmp_vars = node.forward_cuda_kernel(cg, num_tmp_vars, sub_idx_vars, self)
            icode = indent(icode, 4)
            code += """\
if( %(abs_idx)s >= %(offset)d && %(abs_idx)s < %(endoffset)d )
{
    %(sub_idx_var_defs)s

    %(icode)s

    %(res)s = %(var)s;
}""" % locals()
            offset = endoffset
            if idx < len(input_nodes) - 1:
                code += " else "
            else:
                code += "\n"
        return code, res, num_tmp_vars

    def adjoint_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        #print("vstack:adjoint:cuda")
        input_nodes = cg.input_nodes(self)
        found = False
        #idx = input_nodes.index(parent)
        for idx,n in enumerate(input_nodes):
            while isinstance(n, NodeReverseInOut):
                n = n.n
            if n is parent:
                found = True
                break
        assert(found)
        #print("vstack(%s):adjoint -> found parent idx=%d %s sizes=%s" %( repr(self), idx, repr(n), [x.size for x in input_nodes]))
        offset = 0
        for i in range(idx):
            offset += input_nodes[i].size
        shape = self.input_shapes[idx]
        var = "idx_%d" % num_tmp_vars
        num_tmp_vars += 1
        code = ("int %(var)s = %(offset)d + (" % locals()) + sub2ind(abs_idx, shape) + ");\n"
        #print(" called by parent %s, idx=%d -> offset=%d, shape=%s, code=%s" % (n, idx, offset, shape,code))
        try:
            icode, var, num_tmp_vars = cg.output_nodes(self)[0].adjoint_cuda_kernel(cg, num_tmp_vars, [var], self)
        except KeyError:
            res = "var_%(num_tmp_vars)d" % locals()
            num_tmp_vars += 1
            icode = "float %(res)s = x[%(var)s];\n" % locals()
            var = res
        return code + icode, var, num_tmp_vars

    def is_gram_diag(self, freq=False):
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?
        """
        return all([arg.is_gram_diag(freq) for arg in self.input_nodes])

    def get_diag(self, freq=False):
        """Returns the diagonal representation (A^TA)^(1/2).

        Parameters
        ----------
        freq : bool
            Is the diagonal representation in the frequency domain?
        Returns
        -------
        dict of variable to ndarray
            The diagonal operator acting on each variable.
        """
        var_diags = {var: np.zeros(var.size) for var in self.variables()}
        for arg in self.input_nodes:
            arg_diags = arg.get_diag(freq)
            for var, diag in arg_diags.items():
                var_diags[var] = var_diags[var] + diag * np.conj(diag)
        # Get (A^TA)^{1/2}
        for var in self.variables():
            var_diags[var] = np.sqrt(var_diags[var])
        return var_diags

    def norm_bound(self, input_mags):
        """Gives an upper bound on the magnitudes of the outputs given inputs.

        Parameters
        ----------
        input_mags : list
            List of magnitudes of inputs.

        Returns
        -------
        float
            Magnitude of outputs.
        """
        return np.linalg.norm(input_mags, 2)


class split(vstack):

    def __init__(self, output_nodes, implem=None):
        self.output_nodes = output_nodes
        self.shape = [node.shape for node in output_nodes]
        self.input_nodes = []
        super(split,self).__init__(output_nodes, implem)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        super(split, self).adjoint(inputs, outputs)

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        super(split, self).forward(inputs, outputs)

    def forward_cuda_kernel(self, cg, num_tmp_variables, abs_idx, parent):
        #print("split:forward:cuda")
        return super(split, self).adjoint_cuda_kernel(ReverseInOut(cg), num_tmp_variables, abs_idx, parent)

    def adjoint_cuda_kernel(self, cg, num_tmp_variables, abs_idx, parent):
        #print("split:adjoint:cuda")
        return super(split, self).forward_cuda_kernel(ReverseInOut(cg), num_tmp_variables, abs_idx, parent)

    def norm_bound(self, input_mags):
        """Gives an upper bound on the magnitudes of the outputs given inputs.

        Parameters
        ----------
        input_mags : list
            List of magnitudes of inputs.

        Returns
        -------
        float
            Magnitude of outputs.
        """
        return input_mags[0]
