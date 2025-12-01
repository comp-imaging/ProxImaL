import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use as use_renderer

from proximal import Problem, Variable, grad, group_norm1, mul_elemwise, sum_squares, warp
from proximal.lin_ops.lin_op import LinOp


def get_affine(image: np.ndarray, shear_factor: float, hscale: float, target_width: int):
    height, width = image.shape
    M_center = np.array(
        [
            [1, 0, -width / 2.0],
            [0, 1, -height / 2.0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    M_shear = np.array(
        [
            [1, shear_factor, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    M_hscale = np.array(
        [
            [hscale, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    M_vscale = np.array(
        [
            [1, 0, 0],
            [0, target_width / height, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    M_center_inv = np.array(
        [
            [1, 0, target_width * 0.5],
            [0, 1, target_width * 0.5],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    return (M_center_inv @ M_vscale @ M_hscale @ M_shear @ M_center).astype(np.float32)


def make_hstack(im: np.ndarray) -> np.ndarray:
    H, W = im.shape

    tiled = np.zeros((H, im.shape[1] * 3), dtype=im.dtype)
    tiled[1:, :W] = im[: H - 1]
    tiled[:, W : W * 2] = im
    tiled[: H - 1, W * 2 : W * 3] = im[1:]

    return tiled


def apply_affine(tiled, shear_factor=1.0, hscale=1.0, target_width=None):
    height, width = tiled.shape

    if target_width is None:
        target_width = width

    M = get_affine(tiled, shear_factor, hscale, target_width)

    return M, cv2.warpAffine(
        tiled,
        M[:2],
        (target_width, target_width),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_NEAREST,
    )


def generate_warp_mask(u: Variable, target_shape: tuple[int, int], M: np.ndarray) -> np.ndarray:
    warp_term = warp(pad(u, target_shape), M)
    input_fov_image = np.zeros(target_shape, dtype=np.float32, order="F")
    input_fov_image[: u.shape[0], : u.shape[1]] = 1.0

    mask = np.empty(target_shape, dtype=np.float32, order="F")
    warp_term.forward([input_fov_image], [mask])

    mask = (mask > 0.5).astype(np.uint8)
    return mask


class pad(LinOp):
    """Add zero-valued pixels at the right and bottom side. This is to work
    around the limition of proximal.warp operator that requires input and output
    dimensions to be identical."""

    def __init__(self, arg: Variable | LinOp, target_shape=tuple[int, int]):
        assert arg.shape[0] < target_shape[0]
        assert arg.shape[1] < target_shape[1]

        super(pad, self).__init__([arg], target_shape)

    def forward(self, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
        outputs[0][...] = 0.0

        H, W = inputs[0].shape
        outputs[0][:H, :W] = inputs[0]

    def adjoint(self, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
        H, W = outputs[0].shape
        outputs[0][...] = inputs[0][:H, :W]

    def is_gram_diag(self, freq: bool = False) -> bool:
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?"""
        return not freq and self.input_nodes[0].is_diag(freq)

    def get_diag(self, freq: bool = False):
        raise NotImplementedError("Not necessary for Pixel-SR use cases")

    def norm_bound(self, input_mags: list[float]) -> float:
        return input_mags[0]


def visualize(
    filename: str,
    raw: np.ndarray,
    estimated: np.ndarray,
    recovered: np.ndarray,
    n_cycles: int = 5,
    sampling_rate: float = 5e9,
) -> None:
    time = np.arange(0, raw.size) / np.float32(sampling_rate)

    plt.figure(1, figsize=(10, 6))

    plt.subplot(311)
    plt.plot(time * 1e3, raw.ravel(), label="Raw data")
    plt.legend()
    plt.xlabel("Time / ms")

    plt.subplot(312)
    plt.stairs(
        raw[:n_cycles].ravel(),
        time[: raw.shape[1] * n_cycles + 1] * 1e9,
        label="First 5 raster scan lines",
    )
    plt.legend()
    plt.xlabel("Time / ns")
    plt.ylabel("Photodetector readout / steps")

    plt.subplot(313)
    plt.stairs(
        raw[:1].ravel(),
        time[: raw.shape[1] + 1] * 1e9,
        label="One raster scan line",
    )
    plt.legend()
    plt.xlabel("Time / ns")

    plt.tight_layout()
    plt.savefig(f"{filename}-time-trace.png")

    plt.figure(2, figsize=(10, 4))

    plt.subplot(131)
    plt.imshow(
        raw,
        cmap="gray",
        extent=[0, time[raw.shape[1]] * 1e9, 0, time[-1] * 1e3],
        aspect=1e3,
    )
    plt.xlabel("Time / ns")
    plt.ylabel("Time / ms")
    plt.title("Raw data,\nrearranged in 2D")

    plt.subplot(132)
    plt.imshow(estimated, cmap="gray")
    plt.axis("off")
    plt.title("Raw data,\nskew corrected")

    plt.subplot(133)
    plt.imshow(recovered, cmap="gray")
    plt.axis("off")
    plt.title("Recovered by ProxImaL")

    plt.tight_layout()
    plt.savefig(f"{filename}-results.png")


if __name__ == "__main__":
    use_renderer("AGG")

    with h5py.File("data/oil_emulsion.hdf5", "r") as f:
        width = f.attrs["pulse_interval"]
        sample_count = f["raw"].shape[0]
        height = sample_count // width

        raw = f["raw"][()].reshape((height, width))

        shear_factor = f.attrs["shear_factor"]

    oversample_factor = 12.0
    target_width = 256
    tiled = make_hstack(raw)
    M, estimated = apply_affine(tiled, shear_factor, oversample_factor, target_width)
    background = np.median(estimated[:200], axis=0)

    u = Variable((target_width, target_width))
    mask_fov = generate_warp_mask(u, tiled.shape, M)

    # Exclude saturated pixels
    mask = mask_fov * (tiled < tiled.max())

    # Apply mask
    b = np.asfortranarray(tiled, dtype=np.float32)
    b[mask < 0.5] = 0.0
    scaling_factor = b.max()
    b /= scaling_factor

    # Tile background profile vertically to match the shape of the reconstructed image
    background_image = np.empty(u.shape, dtype=np.float32, order="F")
    background_image[...] = background[np.newaxis, :]
    background_image[...] /= scaling_factor

    problem = Problem(
        [
            sum_squares(
                mul_elemwise(
                    mask,
                    warp(
                        pad(u + background_image, b.shape),
                        M,
                    ),
                )
                - b
            ),
            5e-3 * group_norm1(grad(u, implem="halide"), group_dims=[2], implem="halide"),
        ]
    )

    problem.solve(
        solver="ladmm",
        eps_abs=1e-4,
        eps_rel=1e-3,
        max_iters=300,
        verbose=1,
    )
    visualize("pixel-sr", raw, estimated, u.value)
