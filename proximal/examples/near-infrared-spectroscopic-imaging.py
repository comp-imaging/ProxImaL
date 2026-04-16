import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct

from proximal import Problem, Variable, grad, nonneg, norm1, sum_squares
from proximal.lin_ops.black_box import LinOpFactory


def getDctOp(n_samples, n_modes):
    """Generate the linear operator corresponding to
    inverse-DCT 1D transform.

    Given the coefficients, apply inverse DCT transform to obtain the curve
    representing the smooth signal baseline. Crop to the middle 1/3 of the
    signal.

    To recover the DCT coefficients from the coefficients, pad the input signal
    by 3 times of the range. Resolve boundary conditions with mirrored version
    of the same signal. Then, compute the forward DCT transform to extract the
    first few coefficients.
    """

    def fwd(coeff, baseline):
        assert coeff.size < baseline.size
        baseline[:] = idct(coeff, n=n_samples * 2, norm="ortho")[:n_samples]

    def adj(baseline, coeff):
        assert coeff.size < baseline.size
        mirror_image = np.empty((n_samples * 2,), dtype=np.float32)
        mirror_image[:n_samples] = baseline
        mirror_image[n_samples:] = baseline[::-1]

        coeff[:] = dct(mirror_image, norm="ortho")[: coeff.size]

    return LinOpFactory(
        (n_modes,),
        (n_samples,),
        fwd,
        adj,
        norm_bound=None,
    ), fwd


def separate(raw, alpha=0.2, beta=0.8, n_modes=128):
    intensity_range = raw.max() - raw.min()

    u = Variable((raw.size,))
    v = Variable((n_modes,))

    dct_op, dct_fwd = getDctOp(raw.size, n_modes)
    problem = Problem(
        [
            sum_squares(-u + dct_op(v) - (raw - raw.min()) / intensity_range),
            norm1(grad(u)) * alpha * beta,
            norm1(u) * alpha * (1.0 - beta),
            nonneg(u),
        ]
    )
    problem.solve(
        solver="ladmm",
        verbose=1,
    )

    baseline = np.zeros(u.shape, dtype=np.float32)
    dct_fwd(v.value, baseline)
    return u.value * intensity_range, baseline * intensity_range + raw.min()


if __name__ == "__main__":
    with h5py.File("data/NIRS-data.h5", "r") as f:
        raw = f["raw"][()].astype(np.float32) * f.attrs["scale"]
        t = np.arange(raw.size) / f.attrs["sample_rate"]
    spikes, baseline = separate(raw)

    plt.figure(1, (10, 8))

    plt.subplot(311)
    plt.plot(t, raw, "+", color="gray", label="raw")
    plt.plot(t, -spikes + baseline, "r", label="spikes + baseline")
    plt.plot(t, baseline, "b", label="baseline")
    plt.ylim([raw.min(), raw.max()])
    plt.legend()

    plt.subplot(312)
    plt.plot(t, -spikes, "r")
    plt.ylabel("Near-Infrared spectroscopic imaging (NIRS) response")

    plt.subplot(313)
    plt.plot(t, raw, "+", color="gray", label="raw")
    plt.plot(t, -spikes + baseline, "r", label="spikes + baseline")
    plt.plot(t, baseline, "b", label="baseline")
    plt.ylim([raw.min(), raw.max() - 2e-4])
    plt.xlim([500, 1100])

    plt.xlabel("time / second")
    plt.tight_layout()
    plt.savefig("NIRS-results.png")
