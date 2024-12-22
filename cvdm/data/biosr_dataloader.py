from typing import Iterator, Tuple

import numpy as np


class BioSRDataloader:
    def __init__(
        self,
        path: str,
        n_samples: int,
        im_size: int,
    ) -> None:
        self._x = np.load(f"{path}")['x']
        self._y = np.load(f"{path}")['y']
        self._im_size = im_size
        self._n_samples: int = self._x.shape[0]

    def __len__(self) -> int:
        return self._n_samples

    def get_channels(self) -> Tuple[int, int]:
        return self._x.shape[-1], self._y.shape[-1]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self._x[idx], self._y[idx]

        return x, y

    def __call__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)
