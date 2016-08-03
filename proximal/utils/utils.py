from __future__ import print_function, division

# Imports
import numpy as np
from numpy.fft import fftn, ifftn, fft2, ifft2
import cv2
import timeit
import sys

###############################################################################
# Implementations supported
###############################################################################

Impl = {'numpy': 0, 'halide': 1}

###############################################################################
# TODO: DIRTY HACK FOR BACKWARDS COMPATIBILITY!
###############################################################################

try:
    np.stack(np.array([1]))
except:
    from numpy.core import numeric

    def _stack(arrays, axis=0):
        arrays = [np.asanyarray(arr) for arr in arrays]
        if not arrays:
            raise ValueError('need at least one array to stack')

        shapes = set(arr.shape for arr in arrays)
        if len(shapes) != 1:
            raise ValueError('all input arrays must have the same shape')

        result_ndim = arrays[0].ndim + 1
        if not -result_ndim <= axis < result_ndim:
            msg = 'axis {0} out of bounds [-{1}, {1})'.format(axis, result_ndim)
            raise np.IndexError(msg)
        if axis < 0:
            axis += result_ndim

        sl = (slice(None),) * axis + (numeric.newaxis,)
        expanded_arrays = [arr[sl] for arr in arrays]
        return numeric.concatenate(expanded_arrays, axis=axis)
    np.stack = _stack

###############################################################################
# Image utils
###############################################################################


def im2nparray(img, datatype=np.float32):
    """ Converts and normalizes image in certain datatype (e.g. np.float32) """
    np_img = np.array(img)

    i = np.iinfo(np_img.dtype)
    max_class = 2 ** i.bits - 1
    np_img = np_img.astype(datatype) / max_class

    return np_img

###############################################################################
# Timing utils
###############################################################################

# Store last time stamp globally
global lastticstamp
lastticstamp = []


def tic():
    """ Default timer
    Example: t = tic()
         ... code
         elapsed = toc(t)
         print( '{0}: {1:.4f}ms'.format(message, elapsed) )
    """
    global lastticstamp

    t = timeit.default_timer()
    lastticstamp = t
    return t


def toc(t=[]):
    """ See tic f
    """

    global lastticstamp

    # Last tic
    if not t:
        if lastticstamp:
            t = lastticstamp
        else:
            print('Error: Call to toc did never call tic before.', file=sys.stderr)
            return 0.0

    # Measure time in ms
    elapsed = (timeit.default_timer() - t) * 1000.0  # in ms
    return elapsed

###############################################################################
# FFT utils
###############################################################################


def fftd(I, dims=None):

    # Compute fft
    if dims is None:
        X = fftn(I)
    elif dims == 2:
        X = fft2(I, axes=(0, 1))
    else:
        X = fftn(I, axes=tuple(range(dims)))

    return X


def ifftd(I, dims=None):

    # Compute fft
    if dims is None:
        X = ifftn(I)
    elif dims == 2:
        X = ifft2(I, axes=(0, 1))
    else:
        X = ifftn(I, axes=tuple(range(dims)))

    return X


def circshift(x, shifts):

    for j in range(len(shifts)):
        x = np.roll(x, shifts[j], axis=j)

    return x


def psf2otf(K, outsize, dims=None):

    # Size
    sK = K.shape
    assert len(sK) == len(outsize)

    # Pad to large size and circshift
    padfull = []
    for j in range(len(sK)):
        padfull.append((0, outsize[j] - sK[j]))

    Kfull = np.pad(K, padfull, mode='constant', constant_values=0.0)

    # Circular shift
    shifts = -np.floor_divide(np.array(sK), 2)
    if dims is not None and dims < len(sK):
        shifts = shifts[0:dims]

    Kfull = circshift(Kfull, shifts)

    # Compute otf
    otf = fftd(Kfull, dims)

    # Estimate the rough number of operations involved in the computation of the FFT.
    if dims is not None and dims < len(sK):
        sK = sK[0:dims]

    nElem = np.prod(sK)
    nOps = 0
    for k in range(len(sK)):
        nffts = nElem / sK[k]
        nOps = nOps + sK[k] * np.log2(sK[k]) * nffts

    # Discard the imaginary part of the psf if it's withi roundoff error.
    eps = np.finfo(np.float32).eps
    if np.amax(np.absolute(otf.imag)) / np.amax(np.absolute(otf)) <= nOps * eps:
        otf = otf.real

    return otf

###############################################################################
# Image metrics
###############################################################################


def psnr(x, ref, pad=None, maxval=1.0):

    # Sheck size
    if ref.shape != x.shape:
        raise Exception("Wrong size in PSNR evaluation.")

    # Remove padding if necessary
    if pad is not None:

        ss = x.shape
        il = ()
        for j in range(len(ss)):
            if len(pad) >= j + 1 and pad[j] > 0:
                currpad = pad[j]
                il += np.index_exp[currpad:-currpad]
            else:
                il += np.index_exp[:]

        mse = np.mean((x[il] - ref[il])**2)
    else:
        mse = np.mean((x - ref)**2)

    # MSE
    if mse > np.finfo(float).eps:
        return 10.0 * np.log10(maxval**2 / mse)
    else:
        return np.inf

###############################################################################
# Noise estimation
###############################################################################

# Currently only implements one method
NoiseEstMethod = {'daub_reflect': 0, 'daub_replicate': 1}


def estimate_std(z, method='daub_reflect'):
    # Estimates noise standard deviation assuming additive gaussian noise

    # Check method
    if (method not in NoiseEstMethod.values()) and (method in NoiseEstMethod.keys()):
        method = NoiseEstMethod[method]
    else:
        raise Exception("Invalid noise estimation method.")

    # Check shape
    if len(z.shape) == 2:
        z = z[..., np.newaxis]
    elif len(z.shape) != 3:
        raise Exception("Supports only up to 3D images.")

    # Run on multichannel image
    channels = z.shape[2]
    dev = np.zeros(channels)

    # Iterate over channels
    for ch in range(channels):

        # Daubechies denoising method
        if method == NoiseEstMethod['daub_reflect'] or method == NoiseEstMethod['daub_replicate']:
            daub6kern = np.array([0.03522629188571, 0.08544127388203, -0.13501102001025,
                                  -0.45987750211849, 0.80689150931109, -0.33267055295008],
                                 dtype=np.float32, order='F')

            if method == NoiseEstMethod['daub_reflect']:
                wav_det = cv2.sepFilter2D(z, -1, daub6kern, daub6kern,
                                          borderType=cv2.BORDER_REFLECT_101)
            else:
                wav_det = cv2.sepFilter2D(z, -1, daub6kern, daub6kern,
                                          borderType=cv2.BORDER_REPLICATE)

            dev[ch] = np.median(np.absolute(wav_det)) / 0.6745

    # Return standard deviation
    return dev
