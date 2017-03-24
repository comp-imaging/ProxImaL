from .lin_op import LinOp
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
            
    def init_matlab(self, prefix):
        return "%no code\n"
        
    def forward_matlab(self, prefix, inputs, outputs):
        # matlab uses another way of matrix alignment, so we can't just 
        # do an implicit reshape of our data, but we have to transpose it first
        out = outputs[0]
        res  = "osize = " + "+".join([("numel("+x+")") for x in inputs]) + ";\n"
        res += "o=1;\n"
        res += "%(out)s = zeros([osize, 1], 'single', 'gpuArray');\n" % locals()
        for i in inputs:
            
            res += "%(out)s(o:o+numel(%(i)s)-1) = permute( %(i)s, numel(size(%(i)s)):-1:1 );\n" % locals()
            res += "o = o + numel(%(i)s);\n" % locals()
        return res
         
    def adjoint_matlab(self, prefix, inputs, outputs):
        # matlab uses another way of matrix alignment, so we can't just 
        # do an implicit reshape of our data, but we have to transpose it first
        arg = inputs[0]
        res = "o=1;\n"
        for i,o in enumerate(outputs):
            shape = list(self.input_shapes[i])[::-1]
            perm = list(range(len(shape), 0, -1))
            osize = np.prod(shape)
            res += o + " = permute( reshape(%(arg)s(o:o+%(osize)d-1), %(shape)s), %(perm)s );\n" % locals()
            res += "o = o + %(osize)d;\n" % locals()
        return res
    
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
        
    def forward_matlab(self, prefix, inputs, outputs):
        return super(split, self).adjoint_matlab(prefix, inputs, outputs)

    def adjoint_matlab(self, prefix, inputs, outputs):
        return super(split, self).forward_matlab(prefix, inputs, outputs)

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
