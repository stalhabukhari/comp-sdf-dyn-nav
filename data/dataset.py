from pathlib import Path
from typing import Optional, Union, List

import numpy as np
from rich import print

import data.common as cmn
import data.image_utils as imutils


class RGBDFrameMetadata(cmn.JsonBaseModel):
    rgb_filepath: str
    depth_filepath: str
    transform_c2w: List

    @property
    def transform_c2w_np(self) -> np.ndarray:
        transform_mat = np.array(self.transform_c2w, dtype=np.float32)
        assert transform_mat.shape == (
            4,
            4,
        ), f"Unexpected transform shape: {transform_mat.shape}"
        return transform_mat


class RGBDFrameData:
    """
    A single RGB and corresponding depth frame, accompanied by metadata (such as camera transform)
    """

    def __init__(
        self,
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
        metadata: Optional[RGBDFrameMetadata] = None,
    ):
        """
        :param rgb_img: H,W,3 np.ndarray
        :param depth_img: H,W np.ndarray
        :param metadata: RGBDFrameMetadata
        """
        assert len(rgb_img.shape) == 3 and rgb_img.shape[-1] == 3
        assert len(np.squeeze(depth_img).shape) == 2
        self.rgb_img = rgb_img
        self.depth_img = depth_img
        self.metadata = metadata

    @classmethod
    def from_rgbd_frame_metadata(
        cls, metadata: RGBDFrameMetadata, root_dir: Path = Path("")
    ) -> "RGBDFrameData":
        """
        root_dir: path to directory containing metadata json file
        """
        rgb_filepath = root_dir / metadata.rgb_filepath
        assert (
            rgb_filepath.exists()
        ), f"Could not find rgb image at path: {metadata.rgb_filepath}"
        rgb_img = imutils.load_image_from_disk(filepath=rgb_filepath)
        depth_filepath = root_dir / metadata.depth_filepath
        assert (
            depth_filepath.exists()
        ), f"Could not find depth image at path: {metadata.depth_filepath}"
        depth_img = cmn.load_numpy_array_from_disk(filepath=depth_filepath)
        return cls(rgb_img=rgb_img, depth_img=depth_img, metadata=metadata)

    def save_rgb_img_to_file(self, filepath: Union[str, Path]):
        imutils.save_image_to_disk(self.rgb_img, filepath=filepath)
        self.metadata.rgb_filepath = str(filepath)
        print(f"Saved RGB image: {filepath}")

    def save_depth_img_to_file(self, filepath: Union[str, Path]):
        cmn.save_numpy_array_to_disk(array=self.depth_img, filepath=filepath)
        self.metadata.depth_filepath = str(filepath)
        print(f"Saved depth image: {filepath}")

    def get_transform_c2w_np(self) -> np.ndarray:
        # TODO: Add support for convertion of transform format (e.g., OpenGL)
        assert (
            self.metadata is not None
        ), f"c2w transform matrix is supposed to be in metadata, which is None"
        return self.metadata.transform_c2w_np


class IGibsonDatasetMetadata(cmn.JsonBaseModel):
    """
    This class will be present as a .json file that will be used to load the dataset.
    """

    frames_metadata: List[RGBDFrameMetadata]
    transform_format: Optional[cmn.TransformFormats] = None

    def __len__(self) -> int:
        return len(self.frames_metadata)

    @property
    def transforms_c2w(self) -> np.ndarray:
        return np.array(
            [frame.transform_c2w_np for frame in self.frames_metadata], dtype=np.float32
        )
