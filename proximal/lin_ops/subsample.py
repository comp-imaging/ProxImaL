from .lin_op import LinOp
import numpy as np
from ..utils.cuda_codegen import indent, sub2ind, ind2sub

class subsample(LinOp):
    """Samples every steps[i] pixel along axis i,
       starting with pixel 0.
    """

    def __init__(self, arg, steps):
        self.steps = self.format_shape(steps)
        self.orig_shape = arg.shape
        shape = tuple([(dim - 1) // step + 1 for dim, step in zip(arg.shape, self.steps)])
        super(subsample, self).__init__([arg], shape)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        # Subsample.
        selection = self.get_selection()
        np.copyto(outputs[0], inputs[0][selection])

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        # Fill in with zeros.
        selection = self.get_selection()
        outputs[0][:] *= 0
        outputs[0][selection] = inputs[0]

    def forward_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        #print("subsample:forward:cuda")
        new_abs_idx = list(["(%s)*%d" % (ai, si) if type(ai) is str else ai*si for (ai,si) in zip(abs_idx, self.steps)])
        code, var, num_tmp_vars = cg.input_nodes(self)[0].forward_cuda_kernel(cg, num_tmp_vars, new_abs_idx, self)
        return "/*subsample*/\n"+code, var, num_tmp_vars

    def adjoint_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        #print("subsample:adjoint:cuda")
        resvar = "var_%(num_tmp_vars)d" % locals()
        new_abs_idx = list(["(%s)/%d" % (ai, si) if type(ai) is str else ai//si for (ai,si) in zip(abs_idx, self.steps)])
        pcode, var, num_tmp_vars = cg.output_nodes(self)[0].adjoint_cuda_kernel(cg, num_tmp_vars, new_abs_idx, self)
        pcode = indent(pcode, 4)
        sel = " && ". join( ["((%s %% %d) == 0)" % (ai, si) for (ai,si) in zip(abs_idx, self.steps)] )
        code = """/*subsample*/
float %(resvar)s = 0.0f;
if( %(sel)s )
{
    %(pcode)s;
    %(resvar)s = %(var)s;
}
""" % locals()
        return code, resvar, num_tmp_vars

    def get_selection(self):
        """Return a tuple of slices to index into numpy arrays.
        """
        selection = []
        for step in self.steps:
            selection.append(slice(None, None, step))
        return tuple(selection)

    def is_gram_diag(self, freq=False):
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?
        """
        return not freq and self.input_nodes[0].is_diag(freq)

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
        assert not freq
        var_diags = self.input_nodes[0].get_diag(freq)
        selection = self.get_selection()
        self_diag = np.zeros(self.input_nodes[0].shape)
        self_diag[selection] = 1
        for var in var_diags.keys():
            var_diags[var] = var_diags[var] * self_diag.ravel()
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
        return input_mags[0]


class uneven_subsample(LinOp):
    """Samples unevenly according to the given indices.
    The first dimension of indices must be the same as len(arg.shape).
    Only integer indices are supported yet.

    The following two operators are equivalent:

    (1)
        idx0,idx1 = np.indices(arg2d.shape[0]//s0, arg2d.shape[1]//s1)
        uneven_subsample(arg2d, (idx0*s0, idx1*s1))

    (2)
        subsample(arg2d, (s0,s1))
    """

    def __init__(self, arg, indices):
        indices = np.asarray(indices).astype(np.int32)
        assert(len(arg.shape) == indices.shape[0])
        invalid_indices = np.zeros(indices.shape[1:], dtype=np.bool)
        for d in range(len(arg.shape)):
            invalid_indices = np.logical_or(invalid_indices, indices[d] < 0)
            invalid_indices = np.logical_or(invalid_indices, indices[d] >= arg.shape[d])
        self.invalid_indices = invalid_indices
        self.valid_indices = np.logical_not(invalid_indices)
        self.linear_indices = np.ravel_multi_index(indices, arg.shape, mode='clip')
        self.orig_shape = arg.shape
        # for cuda:
        self.indices = indices
        bindices = -np.ones(int(np.prod(arg.shape)), dtype=np.int32)
        bindices[self.linear_indices[self.valid_indices]] = np.arange(int(np.prod(self.linear_indices.shape)), dtype=np.int32)[self.valid_indices.flatten()]
        self.invindices = bindices

        shape = self.linear_indices.shape
        super(uneven_subsample, self).__init__([arg], shape)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        np.take( inputs[0].astype(outputs[0].dtype), self.linear_indices, out=outputs[0] )
        outputs[0][self.invalid_indices] = 0

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        o = np.zeros(int(np.prod(outputs[0].shape)), dtype=inputs[0].dtype)
        o[self.linear_indices[self.valid_indices]] = inputs[0][self.valid_indices]
        outputs[0][:] = np.reshape(o, outputs[0].shape)

    def cuda_additional_buffers(self):
        return [("uneven_subsample_fidx_%d" % self.linop_id, self.indices), ("uneven_subsample_bidx_%d" % self.linop_id, self.invindices)]

    def forward_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        linidx = sub2ind(abs_idx, self.indices.shape[1:])
        vlinidx = "linidx_%(num_tmp_vars)d" % locals()
        code = "/*uneven_subsample*/\nint %(vlinidx)s = %(linidx)s;\n" % locals()
        num_tmp_vars += 1
        res = "res_%(num_tmp_vars)d" % locals()
        code += "float %(res)s = 0.0f;\n" % locals()
        num_tmp_vars += 1
        vvalid = "valid_%(num_tmp_vars)d" % locals()
        code +="int %(vvalid)s = 1;\n" % locals()
        num_tmp_vars += 1
        idx = []
        new_idx = []
        fidx = "uneven_subsample_fidx_%d" % self.linop_id
        offset = 0
        for i in range(self.indices.shape[0]):
            new_var = "idx_%(num_tmp_vars)d" % locals()
            new_idx.append(new_var)
            num_tmp_vars += 1
            shapedim = self.orig_shape[i]
            code += """
int %(new_var)s = %(fidx)s[%(vlinidx)s + %(offset)d];
%(vvalid)s = %(vvalid)s && %(new_var)s >= 0 && %(new_var)s < %(shapedim)d;
""" % locals()
            offset += int(np.prod(self.indices.shape[1:]))
        icode, var, num_tmp_vars = cg.input_nodes(self)[0].forward_cuda_kernel(cg, num_tmp_vars, new_idx, self)
        icode = indent(icode, 4)
        code += """
if( %(vvalid)s )
{
    %(icode)s
    %(res)s = %(var)s;
}
""" % locals()
        return code, res, num_tmp_vars

    def adjoint_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        linidx = sub2ind(abs_idx, self.orig_shape)

        vlinidx = "linidx_%(num_tmp_vars)d" % locals()
        code = "/*uneven_subsample*/\nint %(vlinidx)s = %(linidx)s;\n" % locals()
        num_tmp_vars += 1

        res = "res_%(num_tmp_vars)d" % locals()
        code += "float %(res)s = 0.0f;\n"  % locals()
        num_tmp_vars += 1

        bidx = "uneven_subsample_bidx_%d" % self.linop_id

        newlinidx = "idx_%(num_tmp_vars)d" % locals()
        code += "int %(newlinidx)s = %(bidx)s[%(vlinidx)s];\n"  % locals()
        num_tmp_vars += 1

        new_idx = ind2sub(newlinidx, self.shape)

        icode, var, num_tmp_vars = cg.output_nodes(self)[0].adjoint_cuda_kernel(cg, num_tmp_vars, new_idx, self)
        icode = indent(icode, 4)
        code += """
if(%(newlinidx)s >= 0)
{
    %(icode)s
    %(res)s = %(var)s;
}
""" % locals()
        return code, res, num_tmp_vars


    def is_gram_diag(self, freq=False):
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?
        """
        return not freq and self.input_nodes[0].is_diag(freq)

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
        assert not freq
        var_diags = self.input_nodes[0].get_diag(freq)
        self_diag = np.zeros(int(np.prod(self.input_nodes[0].shape)))
        self_diag[self.linear_indices] = 1
        self_diag = np.reshape(self_diag, self.input_nodes[0].shape)
        for var in var_diags.keys():
            var_diags[var] = var_diags[var] * self_diag.ravel()
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
        return input_mags[0]
