from .prox_fn import ProxFn
import numpy as np
from proximal.utils.utils import Impl
from proximal.halide.halide import Halide
import cv2


class patch_NLM(ProxFn):
    """The function for NLM patch prior
    """

    def __init__(self, lin_op, sigma_fixed=0.6, sigma_scale=6.0,
                 templateWindowSizeNLM=3, searchWindowSizeNLM=11,
                 gamma_trans=1.0, prior=1, **kwargs):

        # Check for the shape
        if not (len(lin_op.shape) == 2 or len(lin_op.shape) == 3 and lin_op.shape[2] in [1, 3]):
            raise ValueError('NLM needs a 3 or 1 channel image')

        self.sigma_fixed = sigma_fixed
        self.sigma_scale = sigma_scale
        self.templateWindowSizeNLM = templateWindowSizeNLM
        self.searchWindowSizeNLM = searchWindowSizeNLM
        self.gamma_trans = gamma_trans
        self.prior = prior

        # Force regular NLM for grayscale
        if len(lin_op.shape) == 2 or len(lin_op.shape) == 3 and lin_op.shape[2] == 1:
            self.prior = 0

        # Halide
        if len(lin_op.shape) == 3 and lin_op.shape[2] == 3:
            self.tmpout = np.zeros(lin_op.shape, dtype=np.float32, order='F')
            self.paramsh = np.asfortranarray(
                np.array([self.sigma_fixed, 1.0, self.sigma_scale, self.prior],
                         dtype=np.float32)[..., np.newaxis])

        super(patch_NLM, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """x = denoise_gaussian_NLM( tonemap(v), sqrt(1/rho))
        """

        if self.implementation == Impl['halide'] and \
           len(self.lin_op.shape) == 3 and self.lin_op.shape[2] == 3:

            # Halide implementation
            tmpin = np.asfortranarray(v.astype(np.float32))
            Halide('prox_NLM.cpp').prox_NLM(tmpin, 1. / rho, self.paramsh, self.tmpout)
            np.copyto(v, self.tmpout)

        else:

            # Compute sigma
            sigma = np.sqrt(1.0 / rho)

            # Fixed sigma if wanted
            sigma_estim = sigma
            if self.sigma_fixed > 0.0:
                sigma_estim = self.sigma_fixed / 30.0 * self.sigma_scale

            # Params
            print("Prox NLM params are: [sigma ={0} prior={1} sigma_scale={2}]".format(
                sigma_estim, self.prior, self.sigma_scale))

            # Scale d
            v = v.copy()
            v_min = np.amin(v)
            v_max = np.amax(v)
            v_max = np.maximum(v_max, v_min + 0.01)

            # Scale and offset parameters d
            v -= v_min
            v /= (v_max - v_min)

            # Denoising params
            sigma_luma = sigma_estim
            sigma_color = sigma_estim
            if self.prior > 0.5:
                sigma_luma = sigma_estim * 1.3  # NLM color stronger on luma
                sigma_color = sigma_color * 3.0

            # Transform
            if self.gamma_trans != 1.0:
                vuint = np.uint8(np.clip(v**self.gamma_trans * 255.0, 0, 255)
                                 )  # Quadratic tranform to account for gamma
            else:
                # Quadratic tranform to account for gamma
                vuint = np.uint8(np.clip(v * 255.0, 0.0, 255.0))

            if self.prior <= 0.5:

                vdstuint = cv2.fastNlMeansDenoising(vuint, None,
                                                    sigma_luma * 255.0,
                                                    self.templateWindowSizeNLM,
                                                    self.searchWindowSizeNLM)
            else:

                vdstuint = cv2.fastNlMeansDenoisingColored(vuint, None,
                                                           sigma_luma * 255.0, sigma_color * 255.,
                                                           self.templateWindowSizeNLM,
                                                           self.searchWindowSizeNLM)

            # Convert to float and inverse scale
            if self.gamma_trans != 1.0:
                dst = ((vdstuint.astype(np.float32) / 255.0) **
                       (1.0 / self.gamma_trans)) * (v_max - v_min) + v_min
            else:
                dst = vdstuint.astype(np.float32) / 255.0 * (v_max - v_min) + v_min
            np.copyto(v, dst)

        return v

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return np.inf  # TODO: IGNORE FOR NOW

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.sigma_fixed, self.sigma_scale, self.templateWindowSizeNLM,
                self.searchWindowSizeNLM, self.gamma_trans, self.prior]
