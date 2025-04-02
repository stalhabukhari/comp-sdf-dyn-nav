from dataclasses import dataclass
from typing import List

import numpy as np
import trimesh

from data import common as cmn


@dataclass
class SceneProperties:
    scene_center: np.ndarray
    bounds_extent: np.ndarray
    bounds_transform_inv: np.ndarray  # to_origin


class MeshNormParams(cmn.JsonBaseModel):
    mean_vec: List  # 3-vec
    max_norm: float  # float

    def unnormalize_pt3ds(self, pt3ds: np.ndarray) -> np.ndarray:
        """
        un-normalize mesh points from [-1, 1] with zero mean to original state
        (recorded in params)
        """
        pt3ds = pt3ds.copy()
        pt3ds *= self.max_norm
        pt3ds += np.array(self.mean_vec).astype(np.float32).reshape(1, 3)
        return pt3ds

    def normalize_pt3ds(self, pt3ds: np.ndarray) -> np.ndarray:
        """
        normalize using params that normalize mesh to [-1, 1] with zero mean
        """
        pt3ds = pt3ds.copy()
        pt3ds -= np.array(self.mean_vec).astype(np.float32).reshape(1, 3)
        pt3ds /= self.max_norm
        return pt3ds

    @property
    def norm_tform_mat(self) -> np.ndarray:
        offset_mat = np.eye(4)
        offset_mat[:3, 3] = -np.array(self.mean_vec).astype(np.float32)
        rescale_mat = np.eye(4)
        rescale_mat[:3, :3] = np.eye(3) * 1.0 / self.max_norm
        tform = rescale_mat @ offset_mat
        return tform

    @property
    def unnorm_tform_mat(self) -> np.ndarray:
        rescale_mat = np.eye(4)
        rescale_mat[:3, :3] = np.eye(3) * self.max_norm
        offset_mat = np.eye(4)
        offset_mat[:3, 3] = np.array(self.mean_vec).astype(np.float32)
        tform = offset_mat @ rescale_mat
        return tform


def get_scene_props_from_scene_mesh(scene_mesh: trimesh.Trimesh) -> SceneProperties:
    T_extent_to_scene, bounds_extent = trimesh.bounds.oriented_bounds(scene_mesh)
    scene_center = scene_mesh.bounds.mean(axis=0)

    scene_props = SceneProperties(
        scene_center=scene_center,
        bounds_extent=bounds_extent,
        bounds_transform_inv=T_extent_to_scene,
    )
    return scene_props


def get_scene_props_from_npyfiles(
    T_extent_to_scene: np.ndarray, bounds_extent: np.ndarray, scene_center: np.ndarray
) -> SceneProperties:
    scene_props = SceneProperties(
        scene_center=scene_center,
        bounds_extent=bounds_extent,
        bounds_transform_inv=T_extent_to_scene,
    )
    return scene_props
