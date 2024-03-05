from typing import Optional, Sequence, Union

import cupy as cp
import numpy as np
from cupy.fft import fftn, fftshift, ifftn, ifftshift


def easy_fft(data: np.ndarray, axes: Optional[Sequence[int]] = None) -> cp.ndarray:
    """FFT that includes shifting."""
    return fftshift(fftn(ifftshift(data, axes=axes), axes=axes), axes=axes)


def easy_ifft(data: np.ndarray, axes: Optional[Sequence[int]] = None) -> cp.ndarray:
    """Inverse FFT that includes shifting."""
    return ifftshift(ifftn(fftshift(data, axes=axes), axes=axes), axes=axes)


def light_component_sim(
    img: np.ndarray, ps: float, z: float, central_frec: float, sigma: float
) -> np.ndarray:
    wl_samples = np.linspace(0.4, 0.7)
    qe_curve = np.exp(-((wl_samples - central_frec) ** 2) / (2 * sigma**2))
    img_a = FresnelPropagator(img, ps, wl_samples, z)
    img_t = img_a * np.expand_dims(np.expand_dims(qe_curve, -1), -1)
    img_t = np.nan_to_num(img_t, nan=0.0)
    light_comp: np.ndarray = np.mean(img_t, axis=0)
    return light_comp


def FresnelPropagator(
    E0: np.ndarray, ps: float, lambda0: Union[float, np.ndarray], z: float
) -> np.ndarray:
    # Parameters: E0 - initial complex field in x-y source plane => sqrt(I_0)*e^i phi
    #             ps - pixel size in microns
    #             lambda0 - wavelength in nm
    #             z - z-value (distance from sensor to object)
    #             background - optional background image to divide out from

    k = np.fft.fftfreq(E0.shape[0], ps)
    kxx, kyy = np.meshgrid(k, k)
    e_kxx = cp.asarray(np.expand_dims(kxx, 0))
    e_kyy = cp.asarray(np.expand_dims(kyy, 0))
    e_lam = cp.asarray(np.expand_dims(np.expand_dims(lambda0, -1), -1))
    E0 = cp.asarray(np.expand_dims(E0, 0))
    H = fftshift(
        cp.exp(1j * (2 * np.pi / e_lam) * z)
        * cp.exp(1j * np.pi * e_lam * z * (e_kxx**2 + e_kyy**2)),
        axes=(1, 2),
    )
    E0fft = easy_fft(
        E0, axes=(1, 2)
    )  # Centered about 0 since fx and fy centered about 0
    G = H * E0fft
    Ef = easy_ifft(G, axes=(1, 2))  # Output after deshifting Fourier transform
    I: np.ndarray = (cp.abs(Ef) ** 2).get()
    return I
