import h5py
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
from python_bindings.high_res import high_res
from python_bindings.low_res import low_res
from python_bindings.prox_poisson import prox_poisson

from proximal import (
    Problem,
    Variable,
    sum_squares,
    nonneg,
)
from proximal.lin_ops.black_box import LinOpFactory
from proximal.prox_fns.prox_fn import ProxFn

tile_size = 256


def get_siemen(imwidth: int, radius=[0.1, 0.8], frequency=180 / 5):
    """Simulate the Siemen's star resolution target.

    Given the radial spatial frequency of the radial barcode and the inner/outer
    radii, simulate the 2D barcode. Normalize the intensities.
    """
    x = np.linspace(-1, 1, imwidth)
    xx, yy = np.meshgrid(x, x)
    r = xx + yy * 1j

    return ne.evaluate(
        """where( (real(abs(r) - radius0) >= 0) & (real(radius1 - abs(r)) >= 0) &
                    (sin( arctan2(imag(r), real(r)) * f) >= 0), 1, 0)""",
        {
            "r": r,
            "radius0": radius[0],
            "radius1": radius[1],
            "f": frequency,
        },
    )


ground_truth = (0.02j * get_siemen(tile_size) + 1.0).astype(np.complex64)
plt.imshow(ground_truth.imag, cmap="gray")
plt.axis("off")


wavevector_quantized = np.asfortranarray(
    np.array(
        [
            [128, 128],
            [128, 108],
            [147, 128],
            [128, 147],
            [108, 128],
            [108, 108],
            [147, 108],
            [147, 147],
            [108, 147],
            [128, 89],
            [166, 128],
            [128, 166],
            [89, 128],
            [89, 108],
            [108, 89],
            [147, 89],
            [166, 108],
            [166, 147],
            [147, 166],
            [108, 166],
            [89, 147],
            [90, 90],
            [165, 90],
            [165, 165],
            [90, 165],
            [128, 71],
            [184, 128],
            [128, 184],
            [71, 128],
            [71, 109],
            [109, 71],
            [146, 71],
            [184, 109],
            [184, 146],
            [146, 184],
            [109, 184],
            [71, 146],
            [72, 90],
            [90, 72],
            [165, 72],
            [183, 90],
            [183, 165],
            [165, 183],
            [90, 183],
            [72, 165],
            [73, 73],
            [182, 73],
            [182, 182],
            [73, 182],
        ],
        dtype=np.int32,
    ).T
)

K = wavevector_quantized.shape[1]
plt.plot(wavevector_quantized[0, :], wavevector_quantized[1, :], "+")
plt.xlabel("kx")
plt.ylabel("ky")
plt.axis("equal")


with h5py.File("../data/2019-09-19-hiPSC-SYTO24-PhalloidinAF568-1.hdf5", "r") as f:
    pupil = np.sum(f["corrected_pupil"][:, 1, :tile_size, :tile_size], axis=0).T

peak_phase = np.abs(np.angle(pupil)).max()
plt.imshow(np.angle(pupil), cmap="seismic", vmin=-peak_phase, vmax=peak_phase)
plt.title("lens aberration")
plt.axis("off")


def makePtychographicLinOp():
    """Derive a linear operator reprensenting the oblique illumination and
    imaging under low magnification microscope objective.

    Import the Halide-accelerated pipelines representing the forward and
    backward wavefront propagation governed by Fourier optics. Create a wrapper
    around it having the ProxImaL type: BlackBoxLinOp.

    Capture the wavefront error of the imaging optics, and the incident angles
    of the oblique illuminations.
    """

    pupil_mapped = pupil.reshape(-1, order="F").view(np.float32).reshape((2, tile_size, tile_size), order="F")

    assert tile_size == 256
    assert K == 49

    def fwd(input, output):
        assert input.shape[0] == 2
        assert input.shape[1] == tile_size
        assert input.shape[2] == tile_size
        assert input.dtype == np.float32
        assert np.isfortran(input)

        assert output.shape[0] == 2
        assert output.shape[1] == tile_size
        assert output.shape[2] == tile_size
        assert output.shape[3] == K
        assert output.dtype == np.float32
        assert np.isfortran(output)

        low_res(input, pupil_mapped, wavevector_quantized, output)

    def adj(input, output):
        assert input.shape[0] == 2
        assert input.shape[1] == tile_size
        assert input.shape[2] == tile_size
        assert input.shape[3] == K
        assert np.isfortran(input)

        assert output.shape[0] == 2
        assert output.shape[1] == tile_size
        assert output.shape[2] == tile_size
        assert np.isfortran(output)

        high_res(input, pupil_mapped, wavevector_quantized, output)

    return LinOpFactory(
        (2, tile_size, tile_size),
        (2, tile_size, tile_size, K),
        fwd,
        adj,
        norm_bound=1.0,
    ), fwd


ptychography, fwd = makePtychographicLinOp()


def simulateLowRes(ground_truth):
    """Simulate low resolution images.

    Given the ground truth, simulate oblique illuminations. Capture the image
    under low-magnification microscope objectives. Also introduce optical
    abberations due to imperfect imaging optics. Add artificial Gaussian noise
    representing thermal noise of the CMOS image sensor.
    """
    simulated_imlow_phasemap = np.zeros((tile_size, tile_size, K), dtype=np.complex64, order="F")
    fwd(
        ground_truth.reshape(-1, order="F").view(np.float32).reshape((2, tile_size, tile_size), order="F"),
        simulated_imlow_phasemap.reshape(-1, order="F").view(np.float32).reshape((2, tile_size, tile_size, K), order="F"),
    )

    noisy_image = ne.evaluate(
        "real(wavefront * conj(wavefront)) + noise * 5e-3",
        {
            "wavefront": simulated_imlow_phasemap,
            "noise": np.asfortranarray(np.random.randn(tile_size, tile_size, K)),
        },
    ).astype(np.float32)

    return np.clip(noisy_image, 0, None).astype(np.float32)


simulated_lowres = simulateLowRes(ground_truth)

plt.figure(3, (8, 8), dpi=150)
T = int(np.ceil(np.sqrt(K)))
for i in range(21):
    plt.subplot(T, T, i + 1)
    plt.imshow(simulated_lowres[..., i], cmap="gray")
    plt.axis("off")

plt.suptitle("Oblique illuminated Siemen star target (simulated)")
plt.tight_layout()
plt.show()


class poisson_phase_norm(ProxFn):
    """Poisson noise model for complex-valued wavefront map."""

    def __init__(self, lin_op, bp, **kwargs):
        self.bp = np.asfortranarray(bp)
        self.tmpout = np.empty((2, *bp.shape), dtype=np.float32, order="F")

        super(poisson_phase_norm, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """Proximal operator of the poisson noise model for complex-valued
        wavefront map.

        Retrieve the intensity component of the wavefront, and perform the
        real-valued proximal operator of the poisson norm. Restore the phase
        component after that.
        """
        assert np.all(v.shape[1:] == self.bp.shape)
        prox_poisson(v, 1.0 / rho, self.bp, self.tmpout)

        v[:] = self.tmpout

        return v

    def _eval(self, v):
        assert np.all(v.shape[1:] == self.bp.shape)
        vsum = ne.evaluate(
            "sum(real(abs(v)) - bp * log(real(abs(v)) + 1e-9))",
            {
                "v": v,
                "bp": self.bp,
            },
        )

        return vsum

    def get_data(self):
        return [self.bp]


def ptychographicPhaseRetrieval(simulated_lowres, alpha=0.7e-1, gain=1e3):
    assert simulated_lowres.ndim == 3
    assert simulated_lowres.dtype == np.float32
    assert np.isfortran(simulated_lowres)
    assert simulated_lowres.shape[0] == simulated_lowres.shape[1]

    # The simulated noise source is additive Gaussian, but FPM researchers
    # reports a more robust reconstruction with the Poisson signal distortion
    # model. Here, we artifically scale the photon count such that the
    # multiplicative noise is similar to additive noise for transparent samples.
    # gain = 1e3

    # The object is a weak phase object. Simulate a flat field here. We use it
    # to minimize the phase component of the reconstructed image.
    flat_phase = np.zeros((2, tile_size, tile_size), dtype=np.float32, order="F")
    flat_phase[0, ...] = 1.0

    u = Variable((2, tile_size, tile_size))
    problem = Problem(
        [
            poisson_phase_norm(ptychography(u) * gain, simulated_lowres * gain),
            sum_squares(u - flat_phase) * alpha,
            nonneg(u),
        ],
        implem="halide",
        scale=False,
    )

    problem.solve(
        solver="ladmm",
        max_iters=100,
        eps_abs=7e-2,
        eps_rel=1e-3,
        verbose=1,
        conv_check=20,
        lmb=1.0,
    )

    return u.value.reshape(-1, order="F").view(np.complex64).reshape((tile_size, tile_size), order="F")


field = ptychographicPhaseRetrieval(simulated_lowres)


def intensityRange(im):
    vmin = np.percentile(im, 1)
    vmax = np.percentile(im, 99)

    return im, vmin, vmax


plt.figure(4, (8, 8), dpi=150)

plt.subplot(221)
plt.imshow(
    simulated_lowres[..., 0],
    cmap="gray",
)
plt.axis("off")
plt.title("Simulated low resolution image,\nbrightfield")

plt.subplot(222)
plt.imshow(
    np.mean(simulated_lowres[..., :21], axis=2),
    cmap="gray",
)
plt.axis("off")
plt.title("Simulated low resolution image,\nwidefield")

plt.subplot(223)
mine, vmin, vmax = intensityRange(field.imag)
plt.imshow(
    field.imag,
    vmin=vmin,
    vmax=vmax,
    cmap="gray",
)
plt.axis("off")
plt.title("Restored phase")

plt.subplot(224)
theirs, vmin, vmax = intensityRange(ground_truth.imag)
plt.imshow(
    ground_truth.imag,
    vmin=vmin,
    vmax=vmax,
    cmap="gray",
)
plt.axis("off")
plt.title("Ground truth")

plt.suptitle("Fourier Phychographic reconstruction of weak phase objects")
