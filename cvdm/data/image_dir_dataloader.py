from typing import Iterator, Tuple

import numpy as np
from PIL import Image

from cvdm.utils.data_utils import read_and_patch_image_from_filename


class ImageDirDataloader:
    def __init__(self, paths: np.ndarray, n_samples: int, im_size: int) -> None:
        self._n_samples = min(n_samples, len(paths))
        self._paths = paths[:n_samples]
        self._im_size = im_size

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        filename = self._paths[idx]
        img_patch = read_and_patch_image_from_filename(filename, self._im_size)

        patch_LR = img_patch.resize(
            (self._im_size // 4, self._im_size // 4), Image.LANCZOS
        )
        err_HR = patch_LR.resize((self._im_size, self._im_size), Image.LANCZOS)
        img_patch_arr = np.array(img_patch)
        err_HR_arr = np.array(err_HR)
        return (err_HR_arr / 255.0) * 2 - 1, (img_patch_arr / 255.0) * 2 - 1

    def __call__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)
