from future import annotations 
import numpy as np 
from numpy.typing import NDArray


def normalize_01(img: NDArray[np.generic]) -> NDArray[np.float32]:
    arr = np.asarray(img, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)

    return (arr - vmin) / (vmax - vmin)


def normalize_uint8(img: NDArray[np.generic]) -> NDArray[np.uint8]:
    norm = normalize_01(img)
    return np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)