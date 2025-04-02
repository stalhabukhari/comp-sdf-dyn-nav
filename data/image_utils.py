import io
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def cvt_img_float_to_uint8(img: np.ndarray) -> np.ndarray:
    img = (img * 255).astype(np.uint8)
    return img


def save_image_to_disk(img: np.ndarray, filepath: Union[str, Path]):
    """
    Working with RGB images only.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    assert filepath.parent.exists(), f"Invalid save destination: {filepath}"
    assert (
        filepath.suffix.lower() == ".jpg"
    ), f"Invalid RGB image format: {filepath.suffix}"
    assert (
        img.shape[-1] == 3
    ), f"Expected 3 channels for RGB image, found: {img.shape[-1]}"
    mode = "RGB"
    Image.fromarray(img, mode).save(filepath)


def load_image_from_disk(filepath: Union[str, Path]) -> np.ndarray:
    """
    Working with RGB images only.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    assert filepath.exists(), f"Invalid file path: {filepath}"
    assert (
        filepath.suffix.lower() == ".jpg"
    ), f"Invalid RGB image format: {filepath.suffix}"
    with Image.open(filepath, mode="r") as im:
        img = np.asarray(im)
    return img


def get_img_arr_from_mpl_fig(fig: plt.figure) -> np.ndarray:
    # TODO: Should this be in vis_utils?
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw")
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()
    return img_arr.copy()


def cvt_transform_to_opengl_format(
    transform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert camera-space-2-world-space transform into OpenGL format (for iSDF)
    """
    if transform is None:
        transform = np.eye(4)
    T = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(np.pi), -np.sin(np.pi), 0],
            [0, np.sin(np.pi), np.cos(np.pi), 0],
            [0, 0, 0, 1],
        ]
    )
    return transform @ T


def rescale_intensity_min_max(img: np.ndarray, max_value: float) -> np.ndarray:
    """
    scale an image to [0, 1] range, using the provided min_value, max_value.
    """
    img = img / max_value
    assert img.min() >= 0 and img.max() <= 1, f"image intensities are not in [0, 1]"
    return img


def undo_rescale_intensity_min_max(
    img: np.ndarray, min_value: float, max_value: float
) -> np.ndarray:
    """
    undo scaling done by `rescale_intensity_min_max` function. The parameters should not change.
    """
    img = img * max_value
    # img = img + min_value
    assert (
        img.min() >= min_value and img.max() <= max_value
    ), f"image intensities are not in [{min_value}, {max_value}]"
    return img


class DepthImageNormalize:
    """
    Class wrapper for `rescale_intensity_min_max`
    """

    def __init__(self, max_value: float):
        self.max_value = max_value

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        return rescale_intensity_min_max(img=depth, max_value=self.max_value)


class DepthImageUnNormalize:
    """
    Class wrapper for `undo_rescale_intensity_min_max`
    """

    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        return undo_rescale_intensity_min_max(
            img=depth, min_value=self.min_value, max_value=self.max_value
        )


class DepthFilter:
    """
    Remove depth value greater than a threshold
    """

    def __init__(self, max_depth_thresh: float, assign_out_of_bound: float = 0.0):
        self.max_depth_thresh = max_depth_thresh
        self.assign_out_of_bound = assign_out_of_bound

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        far_mask = depth > self.max_depth_thresh
        depth[far_mask] = self.assign_out_of_bound  # 0.
        return depth


class AddChannelsDim:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x, axis=0)
