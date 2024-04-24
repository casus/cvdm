from typing import Iterator, Tuple

import numpy as np
from skimage.transform import resize

from cvdm.utils.phase_utils import FresnelPropagator, light_component_sim

DZ = -1


class PhasePolychromeDataloader:

    def __init__(
        self,
        path: str,
        n_samples: int,
        im_size: int,
        training: bool,
    ) -> None:

        self._x = np.load(f"{path}/y.npy", mmap_mode="r+")[:n_samples]
        self._n_samples: int = min(n_samples, self._x.shape[0])
        self._training = training
        self._im_size = im_size

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x = self._x[idx][:, :, 0]
        x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        x = np.nan_to_num(x, nan=0.0)
        x = resize(x, (self._im_size, self._im_size)) * 3.5

        if self._training:
            lam_p = np.random.uniform(0, 0.3)
            dz_r = np.random.uniform(0.01, 3)
            dz_g = np.random.uniform(0.01, 3)
            dz_b = np.random.uniform(0.01, 3)

            cen_r = 0.630
            cen_b = 0.450
            cen_g = 0.550

            rnd_r = np.random.uniform(0.01, 0.1)
            rnd_g = np.random.uniform(0.01, 0.1)
            rnd_b = np.random.uniform(0.01, 0.1)

            img_r = light_component_sim(np.exp(1j * x), 0.21, dz_r, cen_r, rnd_r)

            img_r = img_r + np.random.normal(lam_p, lam_p, size=img_r.shape)
            img_g = light_component_sim(np.exp(1j * x), 0.21, dz_g, cen_g, rnd_g)
            img_g = img_g + np.random.normal(lam_p, lam_p, size=img_g.shape)

            img_b = light_component_sim(np.exp(1j * x), 0.21, dz_b, cen_b, rnd_b)

            img_b = img_b + np.random.normal(lam_p, lam_p, size=img_b.shape)
        else:
            img_r = FresnelPropagator(np.exp(1j * x), 0.21, 0.630, DZ)
            img_g = FresnelPropagator(np.exp(1j * x), 0.21, 0.530, DZ)
            img_b = FresnelPropagator(np.exp(1j * x), 0.21, 0.450, DZ)

        img_stack = np.stack(
            [img_r / np.amax(img_r), img_g / np.amax(img_g), img_b / np.amax(img_b)],
            axis=-1,
        )
        img_stack = np.squeeze(img_stack * 2 - 1)

        y = (x / 3.5) * 2 - 1

        return img_stack, np.expand_dims(y, 2)

    def __call__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)
