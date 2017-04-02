from .prox_fn import ProxFn
import numpy as np
import itertools
from proximal.utils.utils import Impl
from proximal.halide.halide import Halide


class group_norm1(ProxFn):
    """
    The function || ||x||_g ||_1, with g being a set of dimensions.
    This isotropic group sparsity proximal operator allows for example
    for isotropic TV regularization as a simple example.
    """

    def __init__(self, lin_op, group_dims, **kwargs):
        """Initialize temporary variables for _prox method.
        """
        self.group_dims = group_dims
        self.v_group_norm = np.zeros(lin_op.shape, dtype=float)

        # Temp array for halide
        self.tmpout = None
        if len(lin_op.shape) in [3, 4] and lin_op.shape[-1] == 2 and \
           self.group_dims == [len(lin_op.shape) - 1]:
            self.tmpout = np.zeros((lin_op.shape[0], lin_op.shape[1],
                                    lin_op.shape[2] if (len(lin_op.shape) > 3) else 1, 2),
                                   dtype=np.float32, order='FORTRAN')

        super(group_norm1, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """x = v *  (1 - (1/rho) * 1/||x||_g )_+
        """

        if self.implementation == Impl['halide'] and \
           len(self.lin_op.shape) in [3, 4] and self.lin_op.shape[-1] == 2 and \
           self.group_dims == [len(self.lin_op.shape) - 1]:

            # Halide implementation
            if len(self.lin_op.shape) == 3:
                tmpin = np.asfortranarray(np.reshape(
                    v, (self.lin_op.shape[0], self.lin_op.shape[1], 1, 2)).astype(np.float32))
            else:
                tmpin = np.asfortranarray(v.astype(np.float32))

            Halide('prox_IsoL1.cpp').prox_IsoL1(tmpin, 1.0 / rho, self.tmpout)  # Call
            np.copyto(v, np.reshape(self.tmpout, self.lin_op.shape))

        else:

            # Numpy implementation
            np.multiply(v, v, self.v_group_norm)
            

            # Sum along dimensions and keep dimensions
            orig_s = v.shape
            for d in self.group_dims:
                self.v_group_norm = np.sum(self.v_group_norm, axis=d, keepdims=True)

            # Sqrt
            np.sqrt(self.v_group_norm, self.v_group_norm)

            # Replicate
            tiles = ()
            for d in range(len(orig_s)):
                if d in self.group_dims:
                    tiles += (orig_s[d],)
                else:
                    tiles += (1,)

            self.v_group_norm = np.tile(self.v_group_norm, tiles)
            
            # Thresholded group norm
            with np.errstate(divide='ignore'):
                np.maximum(0.0, 1.0 - (1.0 / rho) * (1.0 / self.v_group_norm), self.v_group_norm)
            
            # Mult
            v *= self.v_group_norm

        return v
    
    def _prox_cuda(self, rho, gen_v, subidx, linidx, res):
        code = """/*group_norm1*/
float %(res)s = 0.0f;
float tmp;
""" % locals()        
        for gd in itertools.product( *(range(self.lin_op.shape[d]) for d in self.group_dims) ):
            newidx = subidx[:]
            for i,d in enumerate(self.group_dims):
                newidx[d] = gd[i]
            v = gen_v(newidx)
            code += """
tmp = %(v)s;
%(res)s += tmp*tmp;
""" % locals()
        v = gen_v([linidx])
        code += """
%(res)s = sqrt(%(res)s);

if( %(res)s > 0.0f )
{
    %(res)s = max(0.0f, 1.0f - (1.0f / (%(res)s * %(rho)s) ));
} else
{
    %(res)s = 0.0f;
}
%(res)s *= %(v)s;
""" % locals()
        return code
    
    def _gen_matlab_indices(self):
        res = []
        idx = [-1] * len(self.v_group_norm.shape)
        for d in self.group_dims:
            idx[d] = 0
        while 1:
            i = ",".join(':' if x < 0 else str(x+1) for x in idx)
            res.append(i)
            # increase index
            endOfIter = False
            for k in range(len(idx)-1, -2, -1):
                if k == -1:
                    endOfIter = True
                    break
                if idx[k] >= 0:
                    idx[k] += 1
                    if idx[k] >= self.v_group_norm.shape[k]:
                        idx[k] = 0
                    else:
                        break
            if endOfIter:
                break
        return res
    
    def _prox_matlab(self, prefix, output_var, rho_var, v_var, *args, **kwargs):
        # we want to do this (example for a group of size 2x2)
        #   tmp = arrayfun(@(v1,v2,v3,v4)(sqrt( v1^2 + v2^2 + v3^2 + v4^2 )), v(:,:,1,1), v(:,:,1,2), v(:,:,2,1), v(:,:,2,2))
        #   o = bsxfun(@(v, gn)(v*max(0, 1 - (1/rho)*(1/gn))), v, tmp)
        
        idx = self._gen_matlab_indices()
        t = len(idx)
        
        res  = "tmp = arrayfun(@(" + ",".join(["v%d"%k for k in range(t)]) + ")"
        res += "sqrt(" + " + ".join(["v%d^2"%k for k in range(t)]) + "), "
        res += ", ".join(["%s(%s)" % (v_var, i) for i in idx]) + ");\n"
        res += "tmp = arrayfun(@(gn, rho) max(0, 1 - (1/(rho*(gn + (gn == 0))))), tmp, %(rho_var)s );\n" % locals()
        res += "%(output_var)s = bsxfun(@(v,gn)(v*gn), %(v_var)s, tmp);\n" % locals()
        return res
    
    def _eval_matlab(self, prefix, output_var, v_var):
        idx = self._gen_matlab_indices()
        t = len(idx)
        
        res  = "%(output_var)s = sum(reshape(arrayfun(@(" % locals() + ",".join(["v%d"%k for k in range(t)]) + ")"
        res += "sqrt(" + " + ".join(["v%d^2"%k for k in range(t)]) + "), "
        res += ", ".join(["%s(%s)" % (v_var, i) for i in idx]) + "), 1, []));\n"        
        return res
    
    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """

        # Square
        vsum = v.copy()
        np.multiply(v, v, vsum)

        # Sum along dimensions and keep dimensions
        for d in self.group_dims:
            vsum = np.sum(vsum, axis=d, keepdims=True)

        # Sqrt
        np.sqrt(vsum, vsum)

        # L1 norm is then sum of norms
        return np.sum(vsum)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.group_dims]


class weighted_group_norm1(group_norm1):
    """The function ||W.*x||_1.
    """

    def __init__(self, lin_op, group_dims, weight, **kwargs):
        self.weight = weight
        super(weighted_group_norm1, self).__init__(lin_op, group_dims, **kwargs)

    def _prox(self, rho, v, it):
        """x = v *  (1 - (|W|/rho) * /||x||_g )_+
        """

        # Square
        np.multiply(v, v, self.v_group_norm)

        # Sum along dimensions and keep dimensions
        orig_s = v.shape
        for d in self.group_dims:
            self.v_group_norm = np.sum(self.v_group_norm, axis=d, keepdims=True)

        # Sqrt
        np.sqrt(self.v_group_norm, self.v_group_norm)

        # Replicate
        tiles = ()
        for d in range(len(orig_s)):
            if d in self.group_dims:
                tiles += (orig_s[d],)
            else:
                tiles += (1,)

        self.v_group_norm = np.tile(self.v_group_norm, tiles)

        # Thresholded group norm
        with np.errstate(divide='ignore'):
            np.maximum(0.0, 1.0 - (np.absolute(self.weight) / rho) *
                       (1.0 / self.v_group_norm), self.v_group_norm)

        # Mult
        idxs = self.weight != 0
        v[idxs] *= self.v_group_norm[idxs]
        return v
    
    def _prox_matlab(self, *args, **kw):
        raise NotImplemented

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return super(weighted_group_norm1, self)._eval(self.weight * v)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.group_dims, self.weight]
