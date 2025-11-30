"""
Denoise 1-D photodetector traces using ADMM-based total-variation (TV)
minimization. The method recovers a nonnegative, piecewise-flat signal that
remains close to the raw measurements while suppressing high-frequency noise via
a TV prior. This process enhances the visibility of sharp photoresponse pulses,
following the signal-processing approach used in Zhong et al., Science (2022),
doi: https://doi.org/10.1126/science.abo7651.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from proximal import Problem, Variable, grad, nonneg, norm1, sum_squares
from proximal.utils.utils import Impl
from matplotlib import use as use_renderer


def shannon_entropy(signal: np.ndarray):
    vmin = signal.min()
    vmax = signal.max()
    n_bins = int(vmax - vmin + 1)
    assert n_bins > 1

    distribution, _ = np.histogram(signal + vmin, n_bins)
    return entropy(distribution, base=2)


def getProblem(
    signal: np.ndarray,
    alpha: float,
) -> tuple[Variable, Problem]:
    x = Variable((N,))
    hl = Impl["halide"]

    return x, Problem(
        [
            sum_squares(x + signal),
            norm1(grad(x, implem=hl), implem=hl) * alpha,
            nonneg(x),
        ]
    )


if __name__ == "__main__":
    # Plot data in PNG format
    use_renderer("AGG")

    print("Reading photoresponse data...")
    with h5py.File("data/qsi-pulse-samples-06ef7503-cf8f-45f8-87fc-3b2f57fcdfdc.hdf5", "r") as file:
        dset = file["raw"]
        N = dset.size
        trace = dset[:N]

    print(f"""n_samples = {trace.size}
    Shannon entropy = {shannon_entropy(trace):0.2f} bits / sample
    """)

    black_level = 20
    peak = 128.0

    x, problem = getProblem(
        (black_level - trace.astype(np.float32)) / peak,
        alpha=4e-1,
    )
    problem.solve(
        solver="ladmm",
        max_iters=500,
        eps_abs=1e-4,
        eps_rel=1e-3,
        verbose=1,
    )

    plt.figure(1, figsize=(10, 8))

    plt.subplot(211)
    plt.plot(
        trace[50:250],
        "+",
        mec="gray",
        label="Raw photodetector readout",
    )
    plt.plot(
        x.value[50:250] * peak + black_level,
        "b",
        label="Recovered pulses",
    )
    plt.legend()

    plt.subplot(212)
    plt.plot(trace, "+", mec="gray")
    plt.plot(x.value * peak + black_level, "b")

    plt.xlabel("time / ticks")
    plt.ylabel("Photoresponse / A.U.")

    plt.tight_layout()
    plt.savefig("qsi-pulses.png")
