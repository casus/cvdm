from typing import Iterator, Tuple

import cv2
import numpy as np

from cvdm.utils.data_utils import read_and_patch_image_from_filename
from cvdm.utils.phase_utils import FresnelPropagator

WL = 0.521
DZ = -1


class Phase2ShotDataloader:

    def __init__(self, paths: np.ndarray, n_samples: int, im_size: int, training: bool):
        self._n_samples = min(n_samples, len(paths))
        self._paths = paths[:n_samples]
        self._im_size = im_size
        self._training = training

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        filename = self._paths[idx]
        img_patch = read_and_patch_image_from_filename(filename, self._im_size)

        img_patch_arr: np.ndarray = np.array(img_patch)

        img_patch_arr = cv2.cvtColor(img_patch_arr, cv2.COLOR_RGB2GRAY)
        x = np.divide(img_patch_arr, 255.0)
        x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        x = np.nan_to_num(x, nan=0.0) * 3.5

        if self._training:
            lam = np.random.uniform(0, 0.2)
            img_f = FresnelPropagator(np.exp(1j * x), 0.21, np.asarray([WL]), DZ)[
                0
            ] + np.random.normal(lam, lam, x.shape)
            img_b = FresnelPropagator(np.exp(1j * x), 0.21, np.asarray([WL]), -DZ)[
                0
            ] + np.random.normal(lam, lam, x.shape)
        else:
            img_f = FresnelPropagator(np.exp(1j * x), 0.21, np.asarray([WL]), DZ)[0]
            img_b = FresnelPropagator(np.exp(1j * x), 0.21, np.asarray([WL]), -DZ)[0]

        img_stack = np.stack((img_f, img_b), axis=-1)
        img_stack = (img_stack - np.amin(img_stack)) / (
            np.amax(img_stack) - np.amin(img_stack)
        )
        img_stack = np.squeeze(img_stack * 2 - 1)
        y = (x / 3.5) * 2 - 1

        return img_stack, np.expand_dims(y, 2)

    def __call__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:

        for i in range(self.__len__()):
            yield self.__getitem__(i)
