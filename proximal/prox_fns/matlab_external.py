from .prox_fn import ProxFn
import numpy as np
import matlab.engine


class matlab_external(ProxFn):
    """The function for matlab prior
    """

    def __init__(self, lin_op, matengine, proxfunc, evalfunc=None, params=None, **kwargs):

        self.matengine = matengine
        self.params = params
        self.proxfunc = proxfunc
        self.evalfunc = evalfunc

        super(matlab_external, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, it, *args, **kwargs):
        """ calls matlab proximal function on v
        """
        assert it is not None

        # Run external matlab method
        proxmethod = getattr(self.matengine, self.proxfunc)

        vmat = matlab.double(v.tolist())
        if self.params is None:
            pres = np.array(proxmethod(vmat, rho, it))
        else:
            paramsmat = matlab.double(self.params.tolist())
            pres = np.array(proxmethod(vmat, rho, it, paramsmat))

        np.copyto(v, pres)
        return v

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """

        if self.evalfunc is None:
            return np.inf
        else:
            evalmethod = getattr(self.matengine, self.evalfunc)

            vmat = matlab.double(v)
            if self.params is None:
                peval = np.float32(evalmethod(vmat))
            else:
                paramsmat = matlab.double(self.params.tolist())
                peval = np.float32(evalmethod(vmat, paramsmat))

            return peval

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.matengine, self.proxfunc, self.evalfunc, self.params]
