import time
from pathlib import Path
from typing import Optional, Union, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import common as cmn
from data import structures as struct
from data import mesh_utils
from data.structures import SampledPointsBatch
from lib_obj_det.object_detection_utils import YoloFramePredictions
from lib_pc_align import farthest_point_sampling
from lib_nn import torch_utils

from infer_single_view import InferenceSingleViewAugSDF


class TrajectoryOptimizerWholeBody:
    """Uses robot body grid points for TrajOpt"""

    def __init__(
        self,
        sdf_infer_map: InferenceSingleViewAugSDF,
        iterations_per_update: int = 5,
        lr_init: float = 1e-1,
        lr_decay: float = 0.99,
        safety_margin: float = 0.05,
        smoothing: float = 0.15,
        robot_extent_ow: float = 0.2,
    ):
        self.sdf_infer_map = sdf_infer_map
        self.device = self.sdf_infer_map.device
        self.iterations_per_update = iterations_per_update
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.safety_margin = safety_margin
        self.smoothing = smoothing
        # robot size
        grid_res = 7
        self.robot_body_grid = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(-robot_extent_ow, robot_extent_ow, grid_res),
                    torch.linspace(-robot_extent_ow, robot_extent_ow, grid_res),
                    torch.linspace(-robot_extent_ow, robot_extent_ow, grid_res),
                ),
                dim=-1,
            )
            .reshape(-1, 3)
            .to(self.device)
        )

    @staticmethod
    def _decompose_line(pts: torch.Tensor):
        start = pts[0].reshape(1, -1).clone()
        start.requires_grad = False
        end = pts[-1].reshape(1, -1).clone()
        end.requires_grad = False
        inters = pts[1:-1].clone()
        inters.requires_grad = True
        return start, inters, end

    @torch.no_grad()
    def get_oriented_robot_bodies_along_traj(
        self, traj_pt3ds: torch.Tensor
    ) -> torch.Tensor:
        """
        robot bodies are oriented along the trajectory tangents
        """
        N = traj_pt3ds.shape[0]
        dirs = traj_pt3ds[1:] - traj_pt3ds[:-1]
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        dirs = torch.cat((dirs, dirs[-1][None]), dim=0)  # add one extra at the end

        x_vecs = (
            torch.tensor([1.0, 0.0, 0.0], device=self.device).reshape(1, 3).repeat(N, 1)
        )
        z_vecs = (
            torch.tensor([0.0, 0.0, 1.0], device=self.device).reshape(1, 3).repeat(N, 1)
        )
        # get angle wrt x-axis
        angles = torch.arccos((dirs * x_vecs).sum(dim=-1))
        angles_sign = (torch.cross(x_vecs, dirs) * z_vecs).sum(dim=-1).sign()
        angles = angles * angles_sign

        rot_mats = torch.eye(4, dtype=torch.float32, device=self.device)
        rot_mats = rot_mats.reshape(1, 4, 4).repeat(N, 1, 1)
        # rotate about z-axis
        rot_mats[:, 0, 0] = torch.cos(angles)
        rot_mats[:, 0, 1] = -torch.sin(angles)
        rot_mats[:, 1, 0] = torch.sin(angles)
        rot_mats[:, 1, 1] = torch.cos(angles)

        ones = torch.ones(
            len(self.robot_body_grid), 1, dtype=torch.float32, device=self.device
        )
        robot_grid4d = torch.cat((self.robot_body_grid, ones), dim=-1)
        robot_grid4d = (robot_grid4d[None, :, None] * rot_mats[:, None]).sum(dim=-1)
        robot_grid_or = robot_grid4d[..., :3]  # traj-size x grid-size x 3
        return robot_grid_or

    def optimize_trajectory(
        self,
        obj_det_preds: YoloFramePredictions,
        fs_buffer: struct.RGBDFrameSampleBuffer,
        traj_pt2ds: torch.Tensor,
        z_value: float = 0.3,
    ) -> torch.Tensor:
        # optimize
        N = traj_pt2ds.shape[0]
        device_init = traj_pt2ds.device
        start, inters, end = self._decompose_line(pts=traj_pt2ds.to(self.device))
        opt = torch.optim.Adam([inters], lr=self.lr_init)
        safe_marg = torch.tensor(
            self.safety_margin, dtype=torch.float32, device=self.device
        )
        self.sdf_infer_map.set_frame_sample_for_inference_no_yolo(
            obj_det_preds=obj_det_preds, fs_buffer=fs_buffer
        )

        for it in range(self.iterations_per_update):
            traj_pt2ds = torch.cat((start, inters, end), dim=0)
            traj_pt3ds = torch.cat(
                (
                    traj_pt2ds,
                    torch.zeros(N, 1, dtype=torch.float32, device=self.device)
                    + z_value,
                ),
                dim=-1,
            )

            robot_grid_or = self.get_oriented_robot_bodies_along_traj(
                traj_pt3ds=traj_pt3ds
            )
            traj_pt3ds = traj_pt3ds[:, None, :] + robot_grid_or
            traj_pt3ds = traj_pt3ds.view(-1, 3)
            sdf_values = (
                self.sdf_infer_map.infer_sdf_at_points_with_preset_frame_sample(
                    pt3ds=traj_pt3ds
                )
            )
            diff1 = traj_pt2ds[1:, ...] - traj_pt2ds[:-1, ...]
            diff1 = diff1.square().sum(dim=-1)

            loss = self.smoothing * diff1.mean()
            if sdf_values is not None:
                loss_sdf = torch.exp(-sdf_values) * torch.lt(sdf_values, safe_marg)
                loss_sdf = loss_sdf.view(-1, self.robot_body_grid.shape[0])
                loss_sdf = loss_sdf.mean(dim=-1)  # mean robot-grid at each traj point
                loss = loss + loss_sdf.mean()

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            for pg in opt.param_groups:
                pg["lr"] *= self.lr_decay

        traj_pt2ds = torch.cat((start, inters, end), dim=0)
        traj_pt2ds = traj_pt2ds.detach().clone().to(device_init)
        return traj_pt2ds


class PointCloudMemory:
    def __init__(self, radius: float = 1.5):
        self.radius = radius
        self.pc_in_mem: Optional[torch.Tensor] = None
        self.robot_state: Optional[torch.Tensor] = None
        self.cvf: Optional[struct.CameraViewFrustum] = None
        self.device = torch.device("cuda:0")

    def __len__(self):
        if self.pc_in_mem is None:
            return 0
        return len(self.pc_in_mem)

    def get_pts_in_memory(self) -> torch.Tensor:
        """interface for external methods"""
        return self.pc_in_mem

    def set_robot_state(self, new_state: np.ndarray):
        """
        :param new_state: (x, y) only
        """
        self.robot_state = torch.tensor(
            new_state, dtype=torch.float32, device=self.device
        )

    def set_view_frustum(self, cam_params: struct.CameraParams, c2w_tform: np.ndarray):
        self.cvf = struct.CameraViewFrustum.from_cam_params_c2w_tform(
            cam_params=cam_params, c2w_tform=c2w_tform
        )
        self.cvf.compute_frustum_normals()

    def get_pts_in_robot_radius(self, pc: torch.Tensor) -> torch.Tensor:
        pc2d = pc[:, :2]
        dists = torch.linalg.norm(pc2d - self.robot_state.reshape(-1, 2), dim=-1)

        less_than_idx = dists <= self.radius
        dists = dists[less_than_idx]
        pc = pc[less_than_idx]

        great_than_idx = dists > 0.1
        pc = pc[great_than_idx]
        return pc

    def remove_outdated_points_in_mem(self):
        assert self.pc_in_mem is not None
        self.pc_in_mem = self.get_pts_in_robot_radius(pc=self.pc_in_mem)
        outdated_mask = self.cvf.check_pts_inside_frustum_torch(pts=self.pc_in_mem)
        self.pc_in_mem = self.pc_in_mem[~outdated_mask]

    def pre_update_setup(
        self,
        robot_state_new: np.ndarray,
        cam_params: struct.CameraParams,
        c2w_tform: np.ndarray,
    ):
        """use this before trajectory optimization epoch"""
        # TODO: remove redundancy in args
        self.set_robot_state(new_state=robot_state_new)
        self.set_view_frustum(cam_params=cam_params, c2w_tform=c2w_tform)
        if self.pc_in_mem is not None:
            self.remove_outdated_points_in_mem()

    def update(self, pc_new: torch.Tensor):
        """use this after trajectory optimization epoch
        pc_new contains both the previous memory pc and the new points in view frustum
        """
        pc_new = self.get_pts_in_robot_radius(pc=pc_new)
        self.pc_in_mem = pc_new


class TrajectoryOptimizerBodySDF:
    """Uses robot body SDF for TrajOpt"""

    def __init__(
        self,
        sdf_infer_map: InferenceSingleViewAugSDF,
        body_sdf_latent_code_filepath: Path,
        iterations_per_update: int = 5,
        lr_init: float = 1e-1,
        lr_decay: float = 0.99,
        trunc_dist_sdf: float = 0.05,
        smoothing: float = 0.15,
        pc_filter_radius: float = 0.5,
        pc_memory_radius: float = 1.0,
        sdf_scene_map: Callable = None,
    ):
        """
        :param sdf_infer_map:
        :param body_sdf_latent_code_filepath: DeepSDF's latent code for robot body SDF
        :param iterations_per_update: iterations in a single optimization problem
        :param lr_init:
        :param lr_decay:
        :param trunc_dist_sdf: max value of Robot body's SDF considered
        :param smoothing: weightage for first finite difference of trajectory
        :param pc_filter_radius: used to select near-traj pc points on which SDF is inferred
        """
        self.sdf_infer_map = sdf_infer_map
        self.device = self.sdf_infer_map.device

        self.iterations_per_update = iterations_per_update
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.smoothing = smoothing

        self.trunc_dist_sdf = torch.tensor(
            trunc_dist_sdf, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self.pc_filter_radius = pc_filter_radius

        # robot sdf code
        robot_sdf_latent = torch.load(
            body_sdf_latent_code_filepath, map_location=self.device
        ).reshape(1, 256)
        self.fps_fn = farthest_point_sampling.get_fps_fn()
        self.fps_samples_max_per_obj = 1500

        # define robot sdf map
        _json_files = list(body_sdf_latent_code_filepath.parent.glob("*.json"))
        assert len(_json_files) == 1, (
            f"Encountered {len(_json_files)} .json files in ckpt dir "
            "(should be only one corresponding to robot norm params)"
        )
        norm_params_filepath = _json_files[0]
        params = mesh_utils.MeshNormParams.from_json_file(norm_params_filepath)
        self.tform_normalize = torch.tensor(
            params.norm_tform_mat, dtype=torch.float32, device=self.device
        )
        self.sdf_rescaling_norm = params.max_norm
        ## refactoring to reduce matmuls
        self.sdf_robot_map = lambda queries: self.sdf_infer_map.sdf_obj.decode_sdf(
            queries=queries, latent_vector=robot_sdf_latent
        )
        self.sdf_scene_map = sdf_scene_map

        # memory
        self.pc_memory = PointCloudMemory(radius=pc_memory_radius)

    @staticmethod
    def _decompose_line(pts: torch.Tensor):
        start = pts[0].reshape(1, -1).clone()
        start.requires_grad = False
        end = pts[-1].reshape(1, -1).clone()
        end.requires_grad = False
        inters = pts[1:-1].clone()
        inters.requires_grad = True
        return start, inters, end

    @torch.no_grad()
    def get_robot_body_orientations_along_traj(
        self, traj_pt3ds: torch.Tensor
    ) -> torch.Tensor:
        """
        robot bodies are oriented along the trajectory tangents
        """
        N = traj_pt3ds.shape[0]
        dirs = traj_pt3ds[1:] - traj_pt3ds[:-1]
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        dirs = torch.cat((dirs, dirs[-1][None]), dim=0)  # add one extra at the end

        x_vecs = (
            torch.tensor([1.0, 0.0, 0.0], device=self.device).reshape(1, 3).repeat(N, 1)
        )
        z_vecs = (
            torch.tensor([0.0, 0.0, 1.0], device=self.device).reshape(1, 3).repeat(N, 1)
        )
        # get angle wrt x-axis
        angles = torch.arccos((dirs * x_vecs).sum(dim=-1))
        angles_sign = (torch.cross(x_vecs, dirs) * z_vecs).sum(dim=-1).sign()
        angles = angles * angles_sign

        rot_mats = torch.eye(4, dtype=torch.float32, device=self.device)
        rot_mats = rot_mats.reshape(1, 4, 4).repeat(N, 1, 1)
        # rotate about z-axis (use the inverse of this tform for SDF mapping)
        rot_mats[:, 0, 0] = torch.cos(angles)
        rot_mats[:, 0, 1] = -torch.sin(angles)
        rot_mats[:, 1, 0] = torch.sin(angles)
        rot_mats[:, 1, 1] = torch.cos(angles)

        # return rot_mats
        ## refactoring to reduce matmuls
        idxs = torch.zeros(rot_mats.shape[0], dtype=torch.long)
        norm_rot_mats = (self.tform_normalize.reshape(1, 4, 4)[idxs]).bmm(rot_mats)
        return norm_rot_mats

    def optimize_trajectory(
        self,
        pc: torch.Tensor,
        obj_det_preds: Optional[YoloFramePredictions],
        traj_pt2ds: torch.Tensor,
        z_value: float = 0.3,
        check_traj_coll_free: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, bool]]:
        fps_samples_max = 500

        # filter obs PC along z-axis
        pc = pc[pc[:, 2] <= 0.225 + 0.25]  # upto robot height
        pc = pc[pc[:, 2] > 0.02]  # above from ground

        sdf_cost = True
        if pc.numel() == 0:
            # case: no obj pc in view
            if len(self.pc_memory) > 0:
                # memory non-empty
                pc = self.pc_memory.get_pts_in_memory()
            else:
                # memory empty
                sdf_cost = False
        else:
            # case: obj pc in view
            if pc.numel() > fps_samples_max:
                # obj pc is too big
                pc = self.fps_fn(pc, fps_samples_max)
            if len(self.pc_memory) > 0:
                # memory non-empty
                pc_in_mem = self.pc_memory.get_pts_in_memory()
                pc_in_mem = self.fps_fn(pc_in_mem, fps_samples_max * 2)
                pc = torch.cat((pc, pc_in_mem), dim=0)

        # ----------------------------------- optimization -----------------------------------
        N = traj_pt2ds.shape[0]
        device_init = traj_pt2ds.device
        start, inters, end = self._decompose_line(pts=traj_pt2ds.to(self.device))
        opt = torch.optim.Adam([inters], lr=self.lr_init)

        for it in range(self.iterations_per_update):
            traj_pt2ds = torch.cat((start, inters, end), dim=0)

            # smoothing penalty
            diff1 = traj_pt2ds[1:, ...] - traj_pt2ds[:-1, ...]
            diff1 = diff1.square().sum(dim=-1)
            loss = self.smoothing * diff1.mean()

            if sdf_cost or self.sdf_scene_map is not None:
                # need traj points as 3D
                traj_pt3ds = torch.cat(
                    (
                        traj_pt2ds,
                        torch.zeros(N, 1, dtype=torch.float32, device=self.device)
                        + z_value,
                    ),
                    dim=-1,
                )

            if sdf_cost:
                # robot sdf penalty
                tforms_orn = self.get_robot_body_orientations_along_traj(
                    traj_pt3ds=traj_pt3ds
                )
                tforms_orn = tforms_orn.transpose(1, 2)

                pt3ds = pc[None, :, :] - traj_pt3ds[:, None, :]  # MxNx3
                ones = torch.ones(
                    *pt3ds.shape[:2], 1, dtype=torch.float32, device=pt3ds.device
                )
                pt4ds = torch.cat((pt3ds, ones), dim=-1)

                ## refactoring to reduce matmuls
                pc_tformed = (tforms_orn[:, None, ...] * pt4ds[:, :, None, :]).sum(
                    dim=-1
                )
                pc_tformed = pc_tformed[..., :3].view(-1, 3)
                # using subset of points:
                _threshold = self.pc_filter_radius / self.sdf_rescaling_norm
                pc_tformed = pc_tformed[pc_tformed.norm(dim=-1) <= _threshold]

                # TODO: chunked prediction
                # torch_utils.chunked_map()
                sdf_values = self.sdf_robot_map(pc_tformed)  # (MxN)

                loss_sdf = torch.exp(-sdf_values) * torch.lt(
                    sdf_values, self.trunc_dist_sdf
                )
                loss = loss + loss_sdf.mean()

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            for pg in opt.param_groups:
                pg["lr"] *= self.lr_decay

        traj_pt2ds = torch.cat((start, inters, end), dim=0)
        if sdf_cost:
            self.pc_memory.update(pc_new=pc)

        # -------------------------------- check if collision free --------------------------------
        if check_traj_coll_free:
            with torch.no_grad():
                traj_pt3ds = torch.cat(
                    (
                        traj_pt2ds.to(self.device),
                        torch.zeros(N, 1, dtype=torch.float32, device=self.device)
                        + z_value,
                    ),
                    dim=-1,
                )
                tforms_orn = self.get_robot_body_orientations_along_traj(
                    traj_pt3ds=traj_pt3ds
                )
                tforms_orn = tforms_orn.transpose(1, 2)
                pt3ds = pc[None, :, :] - traj_pt3ds[:, None, :]  # MxNx3
                ones = torch.ones(
                    *pt3ds.shape[:2], 1, dtype=torch.float32, device=pt3ds.device
                )
                pt4ds = torch.cat((pt3ds, ones), dim=-1)
                pc_tformed = (tforms_orn[:, None, ...] * pt4ds[:, :, None, :]).sum(
                    dim=-1
                )
                pc_tformed = pc_tformed[..., :3].view(-1, 3)
                _threshold = self.pc_filter_radius / self.sdf_rescaling_norm
                pc_tformed = pc_tformed[pc_tformed.norm(dim=-1) <= _threshold]
                sdf_values = self.sdf_robot_map(pc_tformed)  # (MxN)
                # is_traj_coll_free = (sdf_values > 0).all()
                is_traj_coll_free = (sdf_values > -0.1).all() or (
                    (sdf_values > 0).sum() < 10
                )
            return traj_pt2ds.detach().clone().to(device_init), is_traj_coll_free
        return traj_pt2ds.detach().clone().to(device_init)
