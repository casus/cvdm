from typing import Iterator, Tuple

import numpy as np

from cvdm.utils.data_utils import center_crop, sample_norm_01


class NpyDataloader:
    def __init__(
        self,
        path: str,
        n_samples: int,
        im_size: int,
    ) -> None:
        # TODO: handle the channels better?
        self._x = np.load(f"{path}/x.npy", mmap_mode="r+")[:n_samples][..., 0:1]
        print(self._x.shape)
        self._y = np.load(f"{path}/y.npy", mmap_mode="r+")[:n_samples]
        self._im_size = im_size
        self._n_samples: int = min(n_samples, self._x.shape[0])

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self._x[idx], self._y[idx]

        x = sample_norm_01(center_crop(x, crop_size=2000))
        y = sample_norm_01(center_crop(y, crop_size=2000))
        if x.shape[0] > self._im_size or x.shape[1] > self._im_size:
            center_x = np.random.randint(
                self._im_size // 2, x.shape[1] - self._im_size // 2
            )
            center_y = np.random.randint(
                self._im_size // 2, x.shape[0] - self._im_size // 2
            )

            x = x[
                center_y - self._im_size // 2 : center_y + self._im_size // 2,
                center_x - self._im_size // 2 : center_x + self._im_size // 2,
            ]
            y = y[
                center_y - self._im_size // 2 : center_y + self._im_size // 2,
                center_x - self._im_size // 2 : center_x + self._im_size // 2,
            ]
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
            y = np.expand_dims(y, -1)
        return x, y

    def __call__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)
