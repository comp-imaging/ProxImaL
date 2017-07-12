from ..lin_ops import lin_op
from ..utils.cuda_codegen import sub2ind
import numpy as np

class pxwise_matrixmult(lin_op.LinOp):
    """
    Pixel wise matrix mult:
        
        The constant pixel-wise matrices A (K1 x K2 x ... x Kn x N x M) are multiplied with the pixel wise vectors
        given as argument (K1 x K2 x ... x Kn x M) resulting in pixel wise vectors (K1 x K2 x ... x Kn x N)
    """
    def __init__(self, A, arg, normbound = None):
    
        self.A = np.array(A, np.float32)
        self.normbound = normbound
        
        assert np.all( np.array(self.A.shape[:-2]) == np.array(arg.shape[:-1]) )

        # Set implementation in super-class
        super(pxwise_matrixmult, self).__init__([arg], A.shape[:-1])
        
    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        arg = inputs[0]
        res = np.zeros(self.A.shape[:-1], np.float32)
        for i in range(self.A.shape[-2]):
            for j in range(self.A.shape[-1]):
                res[...,i] += self.A[...,i,j]*arg[...,j]
        np.copyto(outputs[0], res)

    def adjoint(self, outputs, inputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        arg = outputs[0]       
        res = np.zeros(self.A.shape[:-2] + (self.A.shape[-1],), np.float32)
        for i in range(self.A.shape[-1]):
            for j in range(self.A.shape[-2]):
                res[...,i] += self.A[...,j,i]*arg[...,j]
        np.copyto(inputs[0], res)

    def cuda_additional_buffers(self):
        return [("pxwise_matmul_%d" % self.linop_id, self.A)]
        
    def forward_cuda_kernel(self, cg, num_tmp_vars, absidx, parent):
        innode = cg.input_nodes(self)[0]
        A = self.cuda_additional_buffers()[0][0]
        var = "var_%d" % num_tmp_vars
        aidx = "idx_%d" % (num_tmp_vars + 1)
        linaidx = sub2ind(absidx + [0], self.A.shape)
        num_tmp_vars += 2
        code = """/*pxwise_matrixmult*/
float %(var)s = 0.0f;
int %(aidx)s = %(linaidx)s;
""" % locals()
        for i in range(self.A.shape[-1]):
            newidx = absidx[:]
            newidx[-1] = "%d" % i
            icode, ivar, num_tmp_vars = innode.forward_cuda_kernel(cg, num_tmp_vars, newidx, self)
            code += """
%(icode)s
%(var)s += %(ivar)s * %(A)s[%(aidx)s + %(i)d];
""" % locals()
        return code, var, num_tmp_vars

    def adjoint_cuda_kernel(self, cg, num_tmp_vars, absidx, parent):
        innode = cg.output_nodes(self)[0]
        A = self.cuda_additional_buffers()[0][0]
        var = "var_%d" % num_tmp_vars
        aidx = "idx_%d" % (num_tmp_vars + 1)
        linaidx = sub2ind(absidx[:-1] + [0, absidx[-1]], self.A.shape)
        num_tmp_vars += 2
        code = """/*pxwise_matrixmult*/
float %(var)s = 0.0f;
int %(aidx)s = %(linaidx)s;
""" % locals()
        for i in range(self.A.shape[-2]):
            newidx = absidx[:]
            newidx[-1] = "%d" % i
            icode, ivar, num_tmp_vars = innode.adjoint_cuda_kernel(cg, num_tmp_vars, newidx, self)
            aoff = i * self.A.shape[-1]
            code += """
%(icode)s
%(var)s += %(ivar)s * %(A)s[%(aidx)s + %(aoff)d];
""" % locals()
        return code, var, num_tmp_vars

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

        if not self.normbound is None:
            nb = self.normbound
        else:
            # the operator norm of a linear operation given as a matrix A is
            # the largest eigenvalue of (A^T)*A
            # so we have to calculate all eigenvectors of (A^T)*A and take the maximum
            
            ATA = np.zeros(self.A.shape[:-2] + (self.A.shape[-1],self.A.shape[-1]), np.float32)
            for i in range(self.A.shape[-1]):
                for j in range(self.A.shape[-1]):
                    for k in range(self.A.shape[-2]):
                        ATA[...,i,j] += self.A[...,k,i] * self.A[...,k,j]
            
            nb = 0
            if self.A.shape[-2] == 2:
                # eigenvalues for 2x2 matrices [[a,b],[c,d]] can be calculated by
                # lambda = ((a+d) +- sqrt((a+d)**2 - 4*(ad - bc)))/2
                a = ATA[...,0,0]
                b = ATA[...,0,1]
                c = ATA[...,1,0]
                d = ATA[...,1,1]
                gapA = np.sqrt(np.maximum(0, (a+d)**2 - 4*(a*d-b*c)))
                l1 = ((a+d) + gapA)*0.5
                l2 = ((a+d) - gapA)*0.5
                l = np.maximum(np.amax(l1), np.amax(l2))
                nb = max(l,nb)
            else:
                raise NotImplemented
            
            print("pxwise_matrixmult: norm_bound=", nb)
        return nb*input_mags[0]


