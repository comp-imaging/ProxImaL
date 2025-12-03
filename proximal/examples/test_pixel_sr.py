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
    _, width = tiled.shape

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
    warp_term = warp(u, M, target_shape=target_shape, implem="halide")
    input_fov_image = np.ones(u.shape, dtype=np.float32, order="F")

    mask = np.empty(target_shape, dtype=np.float32, order="F")
    warp_term.forward([input_fov_image], [mask])

    mask = (mask > 0.5).astype(np.uint8)
    return mask


def visualize(
    filename: str,
    raw: np.ndarray,
    estimated: np.ndarray,
    recovered: np.ndarray,
    oversample_factor: float,
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
    new_extent = [0, time[estimated.shape[1]] / oversample_factor * 1e9, 0, time[-1] * 1e3]
    plt.imshow(
        estimated,
        cmap="gray",
        extent=new_extent,
        aspect=new_extent[1] / new_extent[3],
    )
    plt.title("Raw data,\nskew corrected")

    plt.subplot(133)
    plt.imshow(
        recovered,
        cmap="gray",
        extent=new_extent,
        aspect=new_extent[1] / new_extent[3],
    )
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

    oversample_factor = 6.0
    target_width = 128
    tiled = make_hstack(raw)
    M, estimated = apply_affine(tiled, shear_factor, oversample_factor, target_width)
    background = np.median(estimated[:200], axis=0)

    u = Variable((target_width, target_width))
    mask_fov = generate_warp_mask(u, tiled.T.shape, M)

    # Exclude saturated pixels
    mask = mask_fov * (tiled.T < tiled.max())

    # Apply mask to the raw data to avoid round off error.
    b = np.asfortranarray(tiled.T, dtype=np.float32)
    b[mask < 0.5] = 0.0
    scaling_factor = b.max()
    b /= scaling_factor

    # Tile background profile vertically to match the shape of the reconstructed image
    background_image = np.empty(u.shape, dtype=np.float32, order="F")
    background_image[...] = background[:, np.newaxis]
    background_image[...] /= scaling_factor

    problem = Problem(
        sum_squares(
            mul_elemwise(
                mask,
                warp(
                    u + background_image,
                    M,
                    target_shape=b.shape,
                    implem="halide",
                )
                - b,
            )
        )
        + 1e-3
        * group_norm1(
            grad(u, implem="halide"),
            group_dims=[2],
            implem="halide",
        )
    )

    problem.solve(
        solver="ladmm",
        eps_abs=2e-5,
        eps_rel=1e-3,
        max_iters=500,
        verbose=1,
    )
    visualize("pixel-sr", raw, estimated, u.value.T, oversample_factor)
