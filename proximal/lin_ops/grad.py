from .lin_op import LinOp
import numpy as np
from proximal.utils.utils import Impl
from proximal.halide.halide import Halide
from ..utils.cuda_codegen import indent


class grad(LinOp):
    """
    gradient operation. can be defined for different dimensions.
    default is n-d gradient.
    """

    def __init__(self, arg, dims=None, implem=None):

        if dims is not None:
            self.dims = dims
        else:
            self.dims = len(arg.shape)  # Full n-d gradient

        shape = arg.shape + (self.dims,)

        # Temp array for halide
        self.tmpfwd = None
        self.tmpadj = None
        if len(arg.shape) in [2, 3] and self.dims == 2:
            self.tmpfwd = np.zeros((arg.shape[0], arg.shape[1],
                                    arg.shape[2] if (len(arg.shape) > 2) else 1, 2),
                                   dtype=np.float32, order='F')

            self.tmpadj = np.zeros((arg.shape[0], arg.shape[1],
                                    arg.shape[2] if (len(arg.shape) > 2) else 1),
                                   dtype=np.float32, order='F')

        super(grad, self).__init__([arg], shape, implem)

    def forward(self, inputs, outputs):
        """The forward operator for n-d gradients.

        Reads from inputs and writes to outputs.
        """

        if self.implementation == Impl['halide'] and \
                (len(self.shape) == 3 or len(self.shape) == 4) and self.dims == 2:
            # Halide implementation
            if len(self.shape) == 3:
                tmpin = np.asfortranarray((inputs[0][..., np.newaxis]).astype(np.float32))
            else:
                tmpin = np.asfortranarray((inputs[0]).astype(np.float32))

            Halide('A_grad.cpp').A_grad(tmpin, self.tmpfwd)  # Call
            np.copyto(outputs[0], np.reshape(self.tmpfwd, self.shape))

        else:

            # Input
            f = inputs[0]

            # Build up index for shifted array
            ss = f.shape
            stack_arr = ()
            for j in range(self.dims):

                # Add grad for this dimension (same as index)
                il = ()
                for i in range(len(ss)):
                    if i == j:
                        il += np.index_exp[np.r_[1:ss[j], ss[j] - 1]]
                    else:
                        il += np.index_exp[:]

                fgrad_j = f[il] - f
                stack_arr += (fgrad_j,)

            # Stack all grads as new dimension
            np.copyto(outputs[0], np.stack(stack_arr, axis=-1))

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """

        if self.implementation == Impl['halide'] and \
                (len(self.shape) == 3 or len(self.shape) == 4) and self.dims == 2:

            # Halide implementation
            if len(self.shape) == 3:
                tmpin = np.asfortranarray(np.reshape(inputs[0],
                                                     (self.shape[0], self.shape[1],
                                                      1, 2)).astype(np.float32))
            else:
                tmpin = np.asfortranarray(inputs[0].astype(np.float32))

            Halide('At_grad.cpp').At_grad(tmpin, self.tmpadj)  # Call
            np.copyto(outputs[0], np.reshape(self.tmpadj, self.shape[:-1]))

        else:

            # Compute comparison (Negative divergence)
            f = inputs[0]

            outputs[0].fill(0.0)
            for j in range(self.dims):

                # Get component
                fj = f[..., j]
                ss = fj.shape

                # Add grad for this dimension (same as index)
                istart = ()
                ic = ()
                iend_out = ()
                iend_in = ()
                for i in range(len(ss)):
                    if i == j:
                        istart += np.index_exp[0]
                        ic += np.index_exp[np.r_[0, 0:ss[j] - 1]]
                        iend_out += np.index_exp[-1]
                        iend_in += np.index_exp[-2]
                    else:
                        istart += np.index_exp[:]
                        ic += np.index_exp[:]
                        iend_in += np.index_exp[:]
                        iend_out += np.index_exp[:]

                # Do the grad operation for dimension j
                fd = fj - fj[ic]
                fd[istart] = fj[istart]
                fd[iend_out] = -fj[iend_in]

                outputs[0] += (-fd)

    def forward_cuda_kernel(self, cg, num_tmp_vars, absidx, parent):
        innode = cg.input_nodes(self)[0]
        idxvars = ["idx_%d" % (num_tmp_vars+d) for d in range(self.dims)]
        var = "var_%d" % (num_tmp_vars+self.dims)
        code = """/*grad*/
"""
        num_tmp_vars += self.dims+1
        newidx = absidx[:-1]
        for d,cidx in enumerate(absidx[:self.dims]):
            selidx = absidx[-1]
            idxvar = idxvars[d]
            ed = self.shape[d]-1
            code += "int %(idxvar)s = ((%(selidx)s) == %(d)d) ? min(%(ed)d, (%(cidx)s)+1) : (%(cidx)s);\n" % locals()
            newidx[d] = idxvar

        icode1,ivar1,num_tmp_vars = innode.forward_cuda_kernel(cg, num_tmp_vars, newidx, self)
        icode2,ivar2,num_tmp_vars = innode.forward_cuda_kernel(cg, num_tmp_vars, absidx[:-1], self)
        code += icode1 + icode2
        code += """
float %(var)s = %(ivar1)s - %(ivar2)s;
""" % locals()
        return code, var, num_tmp_vars

    def adjoint_cuda_kernel(self, cg, num_tmp_vars, absidx, parent):
        innode = cg.output_nodes(self)[0]
        dims = self.dims
        nidx = "idx_%d" % num_tmp_vars
        var = "var_%d" % (num_tmp_vars+1)
        num_tmp_vars += 2
        code = """/*grad*/
float %(var)s = 0.0f;
int %(nidx)s;
""" % locals()
        for d in range(self.dims):
            newidx = absidx[:]
            cidx = absidx[d]
            cd = self.shape[d]-1
            code += "%(nidx)s = %(cidx)s-1\n;" % locals()
            newidx[d] = nidx

            icode1,ivar1,num_tmp_vars = innode.adjoint_cuda_kernel(cg, num_tmp_vars, absidx + [str(d)], self)
            icode2,ivar2,num_tmp_vars = innode.adjoint_cuda_kernel(cg, num_tmp_vars, newidx + [str(d)], self)
            icode1 = indent(icode1, 4)
            icode2 = indent(icode2, 4)

            code += """
if( %(cidx)s == 0 )
{
    %(icode1)s
    %(var)s += %(ivar1)s;
} else if( %(cidx)s == %(cd)s )
{
    %(icode2)s
    %(var)s += -%(ivar2)s;
} else
{
    %(icode1)s
    %(icode2)s
    %(var)s += %(ivar1)s - %(ivar2)s;
}
""" % locals()

        code += "%(var)s = -%(var)s;\n" % locals()
        return code, var, num_tmp_vars

    def get_dims(self):
        """Return the dimensinonality of the gradient
        """
        return self.dims

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
        # 1D gradient operator has spectral norm = 2.
        # ND gradient is permutation of stacked grad in axis 0, axis 1, etc.
        # so norm is 2*sqrt(dims)
        return 2 * np.sqrt(self.dims) * input_mags[0]
