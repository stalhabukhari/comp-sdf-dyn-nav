import copy
from typing import List, Optional

import numpy as np
import torch
from rich import print
from dataclasses import dataclass

from data import dataset as ds
from data import transform as tform
from data import sampling as smp


@dataclass
class CameraParams:
    fx: float
    fy: float
    cx: float
    cy: float
    h: int
    w: int
    # TODO: add distortion coefficients k1, k2, k3, p1, p2


class RGBDFrameSample:
    # TODO: Reduce redundancy of numpy/torch attributes
    def __init__(
        self,
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
        transform: np.ndarray,
        device: torch.device = torch.device("cuda"),
    ):
        # numpy components
        self.rgb_img = rgb_img
        self.depth_img = depth_img
        self.transform = transform
        self.device = device
        self.normals = None
        self.normals_torch = None

    def compute_pc_normals_using_cam_params(self, cam_params: CameraParams):
        if self.normals is not None:
            print(
                "[Warning] Normals for this RGBDFrameSample are already computed. Doing nothing!"
            )
        else:
            depth_img_torch = torch.tensor(self.depth_img).to(self.device)
            pc = tform.pointcloud_from_depth_torch(
                depth=depth_img_torch,
                fx=cam_params.fx,
                fy=cam_params.fy,
                cx=cam_params.cx,
                cy=cam_params.cy,
            )
            self.normals_torch = tform.estimate_pointcloud_normals(points=pc)
            self.normals = self.normals_torch.detach().cpu().numpy()

    @classmethod
    def from_rgbd_frame_data(
        cls, frame_data: ds.RGBDFrameData, device: torch.device
    ) -> "RGBDFrameSample":
        # TODO: avoid this conversion (remove redundancies)
        frame_sample = cls(
            rgb_img=frame_data.rgb_img,
            depth_img=frame_data.depth_img,
            transform=frame_data.get_transform_c2w_np(),
            device=device,
        )
        return frame_sample


@dataclass
class RGBDFrameSampleCollection:
    frames: List[RGBDFrameSample]

    def __len__(self) -> int:
        return len(self.frames)

    def add_frame(self, frame: RGBDFrameSample):
        self.frames.append(frame)

    def remove_frame(self, idx: int):
        self.frames.remove(self.frames[idx])

    def get_rgb_imgs(self) -> np.ndarray:
        frame: RGBDFrameSample
        rgb_imgs: List[np.ndarray] = []
        for frame in self.frames:
            rgb_imgs.append(frame.rgb_img)
        return np.array(rgb_imgs, dtype=frame.rgb_img.dtype)

    def get_rgb_imgs_torch(self, device: torch.device) -> torch.tensor:
        rgb_imgs = self.get_rgb_imgs()
        return torch.tensor(rgb_imgs).to(device)

    def get_depth_imgs(self) -> np.ndarray:
        frame: RGBDFrameSample
        depth_imgs: List[np.ndarray] = []
        for frame in self.frames:
            depth_imgs.append(frame.depth_img)
        return np.array(depth_imgs, dtype=frame.depth_img.dtype)

    def get_depth_imgs_torch(self, device: torch.device) -> torch.tensor:
        depth_imgs = self.get_depth_imgs()
        return torch.tensor(depth_imgs).to(device)

    def get_transforms(self) -> np.ndarray:
        frame: RGBDFrameSample
        transforms: List[np.ndarray] = []
        for frame in self.frames:
            transforms.append(frame.transform)
        return np.array(transforms, dtype=frame.transform.dtype)

    def get_transforms_torch(self, device: torch.device) -> torch.tensor:
        transforms = self.get_transforms()
        return torch.tensor(transforms).to(device)

    def compute_normals_using_cam_params(self, cam_params: CameraParams):
        for frame in self.frames:
            if frame.normals is None:
                # Note that point cloud are computed once (recomputing is not considered)
                frame.compute_pc_normals_using_cam_params(cam_params=cam_params)
                assert frame.normals is not None
                assert frame.normals_torch is not None

    def get_normals_using_cam_params(self, cam_params: CameraParams) -> np.ndarray:
        self.compute_normals_using_cam_params(cam_params=cam_params)
        frame: RGBDFrameSample
        normals: List[np.ndarray] = []
        for frame in self.frames:
            assert frame.normals is not None
            normals.append(frame.normals)
        return np.array(normals, dtype=frame.normals.dtype)

    def get_normals_from_cam_params_torch(
        self, cam_params: CameraParams, device: torch.device
    ) -> torch.tensor:
        normals = self.get_normals_using_cam_params(cam_params=cam_params)
        return torch.tensor(normals).to(device)


@dataclass
class SampledPointsBatch:
    depth_batch: torch.tensor
    pc: torch.tensor
    pc_loc: torch.tensor
    z_vals: torch.tensor
    indices_b: torch.tensor
    indices_h: torch.tensor
    indices_w: torch.tensor
    dirs_C_sample: torch.tensor
    depth_sample: torch.tensor
    T_WC_sample: torch.tensor  # cam-2-world transform
    norm_sample: torch.tensor
    binary_masks: torch.tensor


class RGBDFrameSampleBuffer:
    # TODO: Do we need it in this script? Instead, implement a batch handler
    def __init__(
        self,
        buffer_size: int,
        cam_params: CameraParams,
        dirs_C: np.ndarray,
        depth_min: Optional[float] = None,
    ):
        self.buffer_size = buffer_size
        self.cam_params = cam_params
        self.dirs_C = dirs_C
        self.device = torch.device("cuda")
        self.depth_min = depth_min

        self.frames_coll: RGBDFrameSampleCollection = RGBDFrameSampleCollection(
            frames=[]
        )

    def __len__(self) -> int:
        return len(self.frames_coll)

    def add_frame(self, frame_sample: RGBDFrameSample):
        if len(self.frames_coll) == self.buffer_size:
            # remove oldest frame
            self.frames_coll.remove_frame(idx=0)
        self.frames_coll.add_frame(frame=frame_sample)

    def add_frame_collection(self, frame_coll: RGBDFrameSampleCollection):
        if len(frame_coll) > self.buffer_size:
            raise ValueError(
                f"{len(frame_coll)} > {self.buffer_size} (some samples will go to waste)"
            )
        elif len(frame_coll) == self.buffer_size:
            self.frames_coll = copy.copy(frame_coll)
        else:
            frame_sample: RGBDFrameSample
            for frame in frame_coll.frames:
                self.add_frame(frame_sample=frame)

    def get_surface_points_from_depth_batch(
        self, sampling_mask=None
    ) -> SampledPointsBatch:
        """
        TODO: It may help TEASER if near-surace sample points are also provided.
        """
        num_frames = len(self)
        H, W = self.cam_params.h, self.cam_params.w
        _frame = self.frames_coll.frames[0]
        _shape_check = _frame.depth_img.shape
        assert (H, W) == _shape_check, f"Sanity check fail: {_shape_check}"

        get_masks = True
        depth_batch = self.frames_coll.get_depth_imgs_torch(device=self.device)
        transforms_batch = self.frames_coll.get_transforms_torch(device=self.device)
        normals_batch = self.frames_coll.get_normals_from_cam_params_torch(
            cam_params=self.cam_params, device=self.device
        )
        # Normals will be used to compute loss for gradients
        assert num_frames == depth_batch.shape[0]
        assert num_frames == transforms_batch.shape[0]
        assert num_frames == normals_batch.shape[0]

        # get all points instead of sampling
        indices_b, indices_h, indices_w = smp.sample_pixels_torch(
            n_rays=-1,
            n_frames=num_frames,
            h=H,
            w=W,
            device=self.device,
            sampling_mask=sampling_mask,
        )
        (
            dirs_C_sample,
            depth_sample,
            norm_sample,
            T_WC_sample,
            binary_masks,
            indices_b,
            indices_h,
            indices_w,
        ) = smp.get_batch_data_torch(
            depth_batch=depth_batch,
            T_WC_batch=transforms_batch,
            dirs_C=self.dirs_C,
            indices_b=indices_b,
            indices_h=indices_h,
            indices_w=indices_w,
            norm_batch=normals_batch,
            get_masks=get_masks,
        )

        # convert pixel samples to points
        pc, pc_loc, z_vals = smp.depth_image_to_point_cloud(
            T_WC=T_WC_sample, dirs_C=dirs_C_sample, gt_depth=depth_sample, grad=False
        )
        sample_points = SampledPointsBatch(
            **{
                "depth_batch": depth_batch,
                "pc": pc,
                "pc_loc": pc_loc,
                "z_vals": z_vals,
                "indices_b": indices_b,
                "indices_h": indices_h,
                "indices_w": indices_w,
                "dirs_C_sample": dirs_C_sample,
                "depth_sample": depth_sample,
                "T_WC_sample": T_WC_sample,
                "norm_sample": norm_sample,
                "binary_masks": binary_masks,
            }
        )
        return sample_points

    def samples_points_from_pos_to_depth(
        self, sampling_mask=None
    ) -> SampledPointsBatch:
        """
        get points from current pos to depth (object surface).
        points on surface: occupancy
        points between pos and surface: empty
        for EGO-Planner only
        """
        num_frames = len(self)
        H, W = self.cam_params.h, self.cam_params.w
        _frame = self.frames_coll.frames[0]
        _shape_check = _frame.depth_img.shape
        assert (H, W) == _shape_check, f"Sanity check fail: {_shape_check}"

        get_masks = True
        depth_batch = self.frames_coll.get_depth_imgs_torch(device=self.device)
        transforms_batch = self.frames_coll.get_transforms_torch(device=self.device)
        normals_batch = self.frames_coll.get_normals_from_cam_params_torch(
            cam_params=self.cam_params, device=self.device
        )
        # Normals will be used to compute loss for gradients
        assert num_frames == depth_batch.shape[0]
        assert num_frames == transforms_batch.shape[0]
        assert num_frames == normals_batch.shape[0]

        # get all points instead of sampling
        indices_b, indices_h, indices_w = smp.sample_pixels_torch(
            n_rays=-1,
            n_frames=num_frames,
            h=H,
            w=W,
            device=self.device,
            sampling_mask=sampling_mask,
        )

        (
            dirs_C_sample,
            depth_sample,
            norm_sample,
            T_WC_sample,
            binary_masks,
            indices_b,
            indices_h,
            indices_w,
        ) = smp.get_batch_data_torch(
            depth_batch=depth_batch,
            T_WC_batch=transforms_batch,
            dirs_C=self.dirs_C,
            indices_b=indices_b,
            indices_h=indices_h,
            indices_w=indices_w,
            norm_batch=normals_batch,
            get_masks=get_masks,
        )

        # convert pixel samples to points
        pc, pc_loc, z_vals = smp.sample_along_rays(
            T_WC=T_WC_sample,
            min_depth=0.09,
            max_depth=depth_sample,
            n_stratified_samples=150,
            n_surf_samples=1,
            dirs_C=dirs_C_sample,
            gt_depth=depth_sample,
            grad=False,
        )
        sample_points = SampledPointsBatch(
            **{
                "depth_batch": depth_batch,
                "pc": pc,
                "pc_loc": pc_loc,
                "z_vals": z_vals,
                "indices_b": indices_b,
                "indices_h": indices_h,
                "indices_w": indices_w,
                "dirs_C_sample": dirs_C_sample,
                "depth_sample": depth_sample,
                "T_WC_sample": T_WC_sample,
                "norm_sample": norm_sample,
                "binary_masks": binary_masks,
            }
        )
        return sample_points


@dataclass
class CameraViewFrustum:
    cam_params: CameraParams
    cam_pos: np.ndarray  # 3-array vec
    cam_orn: np.ndarray  # 3x3 rot mat
    frustum_normals: Optional[np.ndarray] = None

    @property
    def tform_c2w(self) -> np.ndarray:
        tform = np.eye(4, dtype=np.float32)
        tform[:3, :3] = self.cam_orn
        tform[:, :3] = self.cam_pos[:].reshape(1, 3)
        return tform

    @classmethod
    def from_cam_params_c2w_tform(
        cls, cam_params: CameraParams, c2w_tform: np.ndarray
    ) -> "CameraViewFrustum":
        cam_orn = c2w_tform[:3, :3]
        cam_pos = c2w_tform[:3, -1].reshape(1, 3)
        return cls(cam_params=cam_params, cam_pos=cam_pos, cam_orn=cam_orn)

    def compute_frustum_normals(self) -> np.ndarray:
        H, W = self.cam_params.h, self.cam_params.w
        fx, fy = self.cam_params.fx, self.cam_params.fy
        cx, cy = self.cam_params.cx, self.cam_params.cy
        R_WC = self.cam_orn

        c = np.array([0, W, W, 0], dtype=np.float32)
        r = np.array([0, 0, H, H], dtype=np.float32)
        x = (c - cx) / fx
        y = (r - cy) / fy
        # direction vectors for corners
        corner_dirs_C = np.vstack((x, y, np.ones(4))).T
        corner_dirs_W = (R_WC * corner_dirs_C[..., None, :]).sum(axis=-1)
        # 3x3 @ 4x1x3 -> 4x3x3

        frustum_normals = np.empty((4, 3), dtype=np.float32)
        frustum_normals[0] = np.cross(corner_dirs_W[0], corner_dirs_W[1])
        frustum_normals[1] = np.cross(corner_dirs_W[1], corner_dirs_W[2])
        frustum_normals[2] = np.cross(corner_dirs_W[2], corner_dirs_W[3])
        frustum_normals[3] = np.cross(corner_dirs_W[3], corner_dirs_W[0])
        frustum_normals = (
            frustum_normals / np.linalg.norm(frustum_normals, axis=1)[:, None]
        )

        self.frustum_normals = frustum_normals
        return frustum_normals

    def check_pts_inside_frustum(self, pts: np.ndarray) -> np.ndarray:
        """
        This returns points inside a frustum of infinite size.
        :param pts: np.ndarray of shape Nx3
        # TODO: add filter for min/max depths
        """
        if self.frustum_normals is None:
            self.compute_frustum_normals()
        cam_center = self.cam_pos
        pts = pts - cam_center
        dots = np.dot(pts, self.frustum_normals.T)
        return (dots >= 0).all(axis=1)

    def check_pts_inside_frustum_torch(self, pts: torch.Tensor) -> torch.Tensor:
        """
        This returns points inside a frustum of infinite size.
        :param pts: np.ndarray of shape Nx3
        # TODO: add filter for min/max depths
        """
        torch_device = pts.device
        if self.frustum_normals is None:
            self.compute_frustum_normals()
        cam_center = torch.tensor(
            self.cam_pos, dtype=torch.float32, device=torch_device
        )
        pts = pts - cam_center.reshape(1, -1)
        frustum_normals = torch.tensor(
            self.frustum_normals.T, dtype=torch.float32, device=torch_device
        )
        dots = torch.matmul(pts, frustum_normals)
        return (dots >= 0).all(dim=1)

    def check_pts_inside_frustum_torch_v2(
        self, pts: torch.Tensor, min_depth: float = 1, max_depth: float = 5
    ) -> torch.Tensor:
        """
        Drafted for EGO-planner implementation
        """
        torch_device = pts.device
        if self.frustum_normals is None:
            self.compute_frustum_normals()
        cam_center = torch.tensor(
            self.cam_pos, dtype=torch.float32, device=torch_device
        )
        pts = pts - cam_center.reshape(1, -1)
        frustum_normals = torch.tensor(
            self.frustum_normals.T, dtype=torch.float32, device=torch_device
        )
        dots = torch.matmul(pts, frustum_normals)
        dists = pts.norm(dim=-1)
        dots[dists < min_depth] = -1
        dots[dists > max_depth] = -1
        return (dots >= 0).all(dim=1)
