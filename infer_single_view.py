# import time
import time

# import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import torch
import tqdm
import trimesh
import matplotlib.pyplot as plt
from rich import print

from data import transform as tform
from data import structures as struct
from data import mesh_utils
from data import common as cmn
from data import image_utils as im_utils
from data import igibson_utils as ig_utils
from lib_obj_det.object_detection_utils import YoloFramePredictions
from lib_pc_align import primitive_pc_align, sdf_pc_align, farthest_point_sampling
from lib_nn import sdf as sdf_nn
from lib_nn import torch_utils
from lib_nn import math_utils
from lib_obj_det import object_detector as obj_det
from lib_vis import vis_mesh_whole_scene as vis_mws
from lib_vis import vis_utils

# plt.switch_backend('agg')


@dataclass
class SDFInferenceParams:
    scale: float
    tform: torch.Tensor


@dataclass
class ObstacleSDFInfo:
    label: int
    bounds: np.ndarray
    inf_param: Optional[SDFInferenceParams] = None

    def is_same_obs(self, obstacle: "ObstacleSDFInfo") -> bool:
        if obstacle.label != self.label:
            return False
        # compute centroid
        dist = np.linalg.norm(
            self.bounds.mean(axis=0).flatten() - obstacle.bounds.mean(axis=0).flatten(),
            ord=2,
        )
        return dist < 0.5


class RobotMemory:
    """
    Obstacles in memory will always be reconstructed in SDF
    """

    def __init__(
        self, max_objs: int = 10, radius_outer: float = 0.2, radius_inner: float = 0.15
    ):
        self.max_objs = max_objs
        self.radius_outer = radius_outer
        self.radius_inner = radius_inner
        self.obs_list: List[ObstacleSDFInfo] = []
        self.robot_state = None

    def __len__(self) -> int:
        return len(self.obs_list)

    def set_robot_state(self, new_state: np.ndarray):
        self.robot_state = new_state

    def dist_from_robot_state(self, state: np.ndarray) -> float:
        return np.linalg.norm(
            self.robot_state.flatten()[:2] - state.flatten()[:2], ord=2
        )

    def obs_dist_from_robot_state(self, obstacle: ObstacleSDFInfo) -> float:
        obs_centroid = obstacle.bounds.mean(axis=0)
        dist = self.dist_from_robot_state(state=obs_centroid)
        return dist

    def add_obs_to_memory(self, obstacle: ObstacleSDFInfo):
        if len(self) == self.max_objs:
            raise Exception(f"Memory buffer full!. Cannot add further")
        self.obs_list.append(obstacle)

    def get_obs_if_in_memory(self, obstacle: ObstacleSDFInfo) -> Optional[int]:
        for idx in range(len(self)):
            if obstacle.is_same_obs(obstacle=self.obs_list[idx]):
                return idx
        return None

    def is_obs_in_freeze_range(self, obstacle: ObstacleSDFInfo) -> bool:
        # check if centroid of bbox of visible pc is in the "static" zone
        dist = self.obs_dist_from_robot_state(obstacle=obstacle)
        if dist <= self.radius_inner:
            return True
        return False

    def is_obs_in_update_range(self, obstacle: ObstacleSDFInfo) -> bool:
        # check if centroid of bbox of visible pc is in the "updating" zone
        dist = self.obs_dist_from_robot_state(obstacle=obstacle)
        if self.radius_inner < dist <= self.radius_outer:
            return True
        return False

    def remove_outdated_obs(self):
        obs_list_updated = []
        for idx in range(len(self)):
            obs = self.obs_list[idx]
            dist = self.obs_dist_from_robot_state(obstacle=obs)
            if dist < self.radius_outer:
                obs_list_updated.append(obs)
        self.obs_list = obs_list_updated

    def update(self, obs_list: Optional[List] = None):
        """
        1. discard outdated obstacles
        2. from a given obstacles list, choose which to add
        NOTE: function note in use right now
        """
        self.remove_outdated_obs()
        if obs_list is None:
            return

        for obs in obs_list:
            if self.is_obs_in_update_range(obstacle=obs):
                idx = self.get_obs_if_in_memory(obstacle=obs)
                if idx is None:
                    self.add_obs_to_memory(obstacle=obs)


class InferenceSingleViewAugSDF:
    def __init__(self, config_dict: Dict, precompute_zero_lvl_bounds: bool = True):
        # no notion of incremental training, keyframes, etc.
        # will train a mini SDF with short-term memory, over a pre-trained static SDF
        self.config_dict = config_dict
        assert torch.cuda.is_available()
        self.device = torch.device("cuda")
        self.fps_fn = farthest_point_sampling.get_fps_fn()
        self.depth_max = 2.0
        self._set_scene_params()
        self._set_directions()
        self._load_dataset()
        self._load_networks()
        self._load_visualizer()
        if precompute_zero_lvl_bounds:
            self._precompute_sdf_object_zero_level_set_bounds()

        memory_config = self.config_dict["memory"]
        self.memory = RobotMemory(
            radius_inner=memory_config["radius_inner"],
            radius_outer=memory_config["radius_outer"],
        )
        # for yolo
        self.__frame_bboxes = YoloFramePredictions(preds=[])

    def _set_scene_params(self):
        raw_scene_mesh_path = Path(self.config_dict["data"]["scene_mesh_filepath"])
        print(f"Loading scene mesh from {raw_scene_mesh_path}")
        assert raw_scene_mesh_path.exists()
        raw_scene_mesh = trimesh.exchange.load.load(
            str(raw_scene_mesh_path), process=False
        )
        scene_props = mesh_utils.get_scene_props_from_scene_mesh(
            scene_mesh=raw_scene_mesh
        )
        self.scene_data_path = None

        self.bounds_transform_inv_torch = (
            torch.from_numpy(scene_props.bounds_transform_inv).float().to(self.device)
        )

    def _set_directions(self):
        camera_config = self.config_dict["data"]["camera"]
        self.cam_params = struct.CameraParams(**camera_config)
        self.dirs_C = tform.ray_dirs_C(
            B=1,
            H=self.cam_params.h,
            W=self.cam_params.w,
            fx=self.cam_params.fx,
            fy=self.cam_params.fy,
            cx=self.cam_params.cx,
            cy=self.cam_params.cy,
            device=self.device,
            depth_type="z",
        )

    def _load_networks(self):
        """
        two networks to load
        - scene SDF (iSDF)
        - object SDF (DeepSDF)
        """
        model_config = self.config_dict["model"]
        model_scale_output = model_config["scale_output"]
        self.ckpt_dir = Path(model_config["ckpt_dir"])

        pos_embed_config = model_config["positional_embedding"]
        pos_embed_num_embeds = pos_embed_config["num_embed_fns"]
        pos_embed_scale_input = pos_embed_config["scale_input"]

        sdf_scene_config = model_config["sdf_scene"]
        sdf_scene_hidden_feat = sdf_scene_config["hidden_feature_size"]
        sdf_scene_hidden_blk = sdf_scene_config["hidden_layers_block"]
        sdf_scene_static_thresh = sdf_scene_config["static_thresh"]
        sdf_scene_ckpt = self.ckpt_dir / sdf_scene_config["ckpt"]

        sdf_obj_config = model_config["sdf_object"]
        sdf_obj_code_len = sdf_obj_config["latent_code_length"]
        sdf_obj_net_specs = sdf_obj_config["network_specs"]
        sdf_obj_ckpt = self.ckpt_dir / sdf_obj_config["ckpt"]
        sdf_obj_code_ckpt = self.ckpt_dir / sdf_obj_config["ckpt_latent"]

        yolo_config = model_config["yolo"]
        yolo_ckpt = self.ckpt_dir / yolo_config["ckpt"]
        yolo_ds_cfg = self.ckpt_dir / yolo_config["dataset_cfg"]
        yolo_conf_thres = yolo_config["conf_thres"]
        yolo_iou_thres = yolo_config["iou_thres"]
        yolo_img_size = yolo_config["img_size"]

        pc_align_config = self.config_dict["pc_aligner"]
        pc_align_opt_iters = pc_align_config["opt_iterations"]
        pc_align_opt_lr_init = pc_align_config["lr_init"]
        pc_align_opt_lr_decay = pc_align_config["lr_decay"]
        self.pc_align_angle_inits = pc_align_config["angle_inits"]
        self.fps_samples_max = 1500  # pc_align_config["fps_samples_max"]

        self.yolo = obj_det.YoloV5Detector(
            weights_filepath=yolo_ckpt,
            dataset_config_filepath=yolo_ds_cfg,
            conf_thres=yolo_conf_thres,
            iou_thres=yolo_iou_thres,
            imgsz=[yolo_img_size, yolo_img_size],
            device=self.device,
        )
        torch_utils.freeze_model(self.yolo.model)

        # positional encoding layer (deterministic)
        self.pose_enc = sdf_nn.PositionalEncoding(
            min_deg=0,
            max_deg=pos_embed_num_embeds,
            scale=pos_embed_scale_input,
            transform=self.bounds_transform_inv_torch,
        ).to(self.device)
        self.pose_enc.eval()

        # scene SDF network (frozen)
        self.sdf_scene = sdf_nn.IsdfMap(
            pos_enc_embedding_size=self.pose_enc.embedding_size,
            hidden_size=sdf_scene_hidden_feat,
            hidden_layers_block=sdf_scene_hidden_blk,
            scale_output=model_scale_output,
        ).to(self.device)
        torch_utils.load_checkpoint(model=self.sdf_scene, ckpt_filepath=sdf_scene_ckpt)
        torch_utils.freeze_model(model=self.sdf_scene)
        self.sdf_scene.eval()

        # sdf map, for filtering points
        self.sdf_stc_pe = lambda _x: self.sdf_scene(self.pose_enc(_x))
        self.sdf_stc_pe_filter = (
            lambda _x: self.sdf_scene(self.pose_enc(_x)) > sdf_scene_static_thresh
        )

        # object SDF network (frozen)
        self.sdf_obj = sdf_nn.DeepSdfDecoderMap(
            latent_size=sdf_obj_code_len, **sdf_obj_net_specs
        ).to(self.device)
        self.sdf_obj.load_state_dict_from_ckpt(filepath=sdf_obj_ckpt)
        torch_utils.freeze_model(model=self.sdf_obj)
        self.sdf_obj.eval()
        self.sdf_obj_latvecs = sdf_nn.LatentCodes(
            ckpt_path=sdf_obj_code_ckpt, device=self.device
        )

        # pc align op
        self.pc_aligner = sdf_pc_align.PcAlignSdfGradOpt(
            iterations=pc_align_opt_iters,
            lr=pc_align_opt_lr_init,
            lr_decay=pc_align_opt_lr_decay,
            device=self.device,
        )

    def _load_dataset(self):
        """
        # train val is just a split for completeness, current code is only for inference.
        """
        self.misc_data = ig_utils.MiscDataHandler(
            misc_config_dict=self.config_dict["data"]["misc"]
        )

    def _load_visualizer(self):
        scene_mesh_filepath = Path(self.config_dict["data"]["scene_mesh_filepath"])
        self.vis = vis_mws.Visualizer(
            config_dict=self.config_dict,
            scene_mesh_file=scene_mesh_filepath,
            scene_data_file=self.scene_data_path,
        )

    def _get_mesh_from_sdf_grid(self, sdf_grid: torch.Tensor) -> trimesh.Trimesh:
        mesh = vis_utils.draw_mesh(
            sdf_grid, self.vis.scene_scale, self.vis.bounds_transform
        )
        return mesh

    def tform_opt_sdf(self, tformed_pts: torch.Tensor, obj_id: int = 0) -> torch.Tensor:
        # improved grad-opt-based
        sdf_map = lambda queries: self.sdf_obj.decode_sdf(
            queries=queries, latent_vector=self.sdf_obj_latvecs[obj_id]
        )
        assert len(self.pc_align_angle_inits) == 1
        rot_ang_init = self.pc_align_angle_inits[0]
        # start_time = time.perf_counter()
        tform_opt = self.pc_aligner.optimize_rotz_tran(
            pt3ds=tformed_pts, sdf_map=sdf_map, rot_ang_init=rot_ang_init
        )
        return tform_opt

    def _precompute_sdf_object_zero_level_set_bounds(self):
        zero_level_sets_filepath = Path("zero_level_sets.pt")
        if zero_level_sets_filepath.exists():
            self.obj_bounds_list = torch.load(
                str(zero_level_sets_filepath), map_location=self.device
            )
        else:
            obj_bounds_list = []
            for obj_id in tqdm.trange(
                len(self.sdf_obj_latvecs),
                desc="Computing bounds for sdf zero-level sets of objects",
            ):
                lat_vec = self.sdf_obj_latvecs[obj_id]
                pc_from_sdf = self.sdf_obj.get_zero_level_set_pc(
                    latent_vec=lat_vec
                )  # on cpu
                pc_from_sdf = pc_from_sdf.to(self.device)

                # pre-process so we don't have to do it in `compute_tform_for_input_pc_via_sdf`
                # 1. unnormalize pc_from_sdf (bring to same scale as iGibson imported object)
                obj_ig_data = self.misc_data.get_ig_object_for_label(label=obj_id)
                params = self.misc_data.get_mesh_norm_params_for_object(
                    object_category=obj_ig_data["name"],
                    object_model=obj_ig_data["instance"],
                )
                pc_from_sdf = math_utils.apply_transform_to_pt3ds(
                    pt3ds=pc_from_sdf,
                    tform=torch.tensor(
                        params.unnorm_tform_mat, dtype=torch.float32, device=self.device
                    ),
                )
                # 2. apply igibson scale to pc_from_sdf: pc_from_sdf *= scale
                scale_mat_np = self.misc_data.compute_igibson_scaling_for_object(
                    object_category=obj_ig_data["name"],
                    object_model=obj_ig_data["instance"],
                )
                scale_mat = torch.tensor(
                    scale_mat_np, dtype=torch.float32, device=self.device
                )
                pc_from_sdf = math_utils.apply_transform_to_pt3ds(
                    pt3ds=pc_from_sdf, tform=scale_mat
                )
                # 3. downsample via fps
                pc_from_sdf = self.fps_fn(pc_from_sdf, self.fps_samples_max)
                pc_from_sdf_bounds = primitive_pc_align.get_pc_bounds(pc=pc_from_sdf)
                # append
                obj_bounds_list.append(pc_from_sdf_bounds)
            torch.save(obj_bounds_list, str(zero_level_sets_filepath))
            self.obj_bounds_list = obj_bounds_list

    def _prefilter_obj_pc_using_prior_info(
        self, pc_input: torch.Tensor, obj_id: int
    ) -> torch.Tensor:
        torch_device = pc_input.device

        # FOR SIM
        obj_ig_data = self.misc_data.get_ig_object_for_label(label=obj_id)

        # pre-filter object pc, based on dimensions
        obj_bbox_extents_scaled = self.misc_data.get_object_bbox_size_scaled(
            object_category=obj_ig_data["name"], object_model=obj_ig_data["instance"]
        )
        max_obj_extent_xy = float(np.sqrt(np.sum(obj_bbox_extents_scaled[:2] ** 2))) * 2

        robot_state = torch.tensor(
            self.memory.robot_state, dtype=torch.float32, device=torch_device
        )
        robot2pc_dists = (robot_state.reshape(1, -1) - pc_input[:, :2]).norm(dim=-1)
        argmin_dist = robot2pc_dists.argmin(dim=0)
        minpt2pc_dists = (
            pc_input[:, :2] - pc_input[argmin_dist, :2].reshape(1, -1)
        ).norm(dim=-1)
        pc_input = pc_input[minpt2pc_dists <= max_obj_extent_xy]
        return pc_input

    def compute_tform_for_input_pc_via_sdf(
        self, pc_input: torch.tensor, obj_id: int
    ) -> Optional[SDFInferenceParams]:
        torch_device = pc_input.device
        obj_ig_data = self.misc_data.get_ig_object_for_label(label=obj_id)

        # check if pc_input is already in memory, or if it needs to be added to memory
        obs_bounds = primitive_pc_align.get_pc_bounds(pc=pc_input)
        obs_sdf_info = ObstacleSDFInfo(
            label=obj_id, inf_param=None, bounds=obs_bounds.cpu().numpy()
        )

        # scenario 1: obstacle is within static/frozen range (ignore)
        if self.memory.is_obs_in_freeze_range(obstacle=obs_sdf_info):
            return None

        # scenario 2: obstacle is in memory-adding range, but maybe in memory already
        if self.memory.get_obs_if_in_memory(obstacle=obs_sdf_info) is not None:
            # static/frozen obstacle, already in memory, no need to compute anything here
            return None

        # Scenario 3: obstacle is not in memory, but maybe in memory-adding range
        pc_from_sdf_subsampled_bounds = self.obj_bounds_list[obj_id]
        # unnormalize pc_from_sdf (bring to same scale as iGibson imported object)
        params = self.misc_data.get_mesh_norm_params_for_object(
            object_category=obj_ig_data["name"], object_model=obj_ig_data["instance"]
        )
        tform_normalize = torch.tensor(
            params.norm_tform_mat, dtype=torch.float32, device=torch_device
        )
        # sdf scaling: get scaling used in un-normalization
        scale_unnorm = params.max_norm

        # apply igibson scale to pc_from_sdf: pc_from_sdf *= scale
        scale_mat_np = self.misc_data.compute_igibson_scaling_for_object(
            object_category=obj_ig_data["name"], object_model=obj_ig_data["instance"]
        )
        scale_mat = torch.tensor(scale_mat_np, dtype=torch.float32, device=torch_device)
        tform_unscale = torch.linalg.inv(scale_mat)
        # sdf scaling: get scaling used in igibson import
        scale_ig = scale_mat_np[0, 0]

        # fps
        pc_input_subsampled = self.fps_fn(pc_input, self.fps_samples_max)

        tform_bbox_center_align = primitive_pc_align.align_bbox_centers(
            pc_align_np=pc_input_subsampled,
            pc_base_np_bounds=pc_from_sdf_subsampled_bounds,
        )
        if tform_bbox_center_align is None:
            return None  # pc is too small to work with

        tform_to_opt_domain = tform_normalize @ tform_unscale @ tform_bbox_center_align
        pc_input_tf_unscaled_normed = math_utils.apply_transform_to_pt3ds(
            pt3ds=pc_input_subsampled, tform=tform_to_opt_domain
        )

        # do sdf-based alignment:
        tform_opted = self.tform_opt_sdf(
            tformed_pts=pc_input_tf_unscaled_normed, obj_id=obj_id
        )

        # full tform will first unscale igibson's scaling, then normalize to deepsdf's domain
        tform = tform_opted @ tform_to_opt_domain

        # obstacle is in memory update range, and tform is computed, hence add to memory
        inf_param = SDFInferenceParams(tform=tform, scale=scale_ig * scale_unnorm)
        if self.memory.is_obs_in_update_range(obstacle=obs_sdf_info):
            obs_sdf_info.inf_param = inf_param
            self.memory.add_obs_to_memory(obstacle=obs_sdf_info)
            return None
        return inf_param

    @staticmethod
    def _get_contour_slice_from_sdf_grid(
        sdf_grid: torch.tensor, slice_num: int = 161, info: str = ""
    ) -> torch.tensor:
        assert sdf_grid.shape[0] == sdf_grid.shape[1] == sdf_grid.shape[2]
        dim = sdf_grid.shape[0]
        y_lims = [0, dim]
        z_lims = [0, dim]
        y_axis = np.linspace(*y_lims, num=dim)
        z_axis = np.linspace(*z_lims, num=dim)
        x = slice_num

        fig, ax = plt.subplots(1, 1)
        sdf_slice = sdf_grid[x, :, :]
        ax.contourf(y_axis, z_axis, sdf_slice, levels=20)
        if info:
            ax.set_title(f"Contour Plot - x={x} ({info})")
        else:
            ax.set_title(f"Contour Plot - x={x}")
        ax.set_xlabel("y-axis")
        ax.set_ylabel("z-axis")

        img = im_utils.get_img_arr_from_mpl_fig(fig=fig)
        plt.close()
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    @staticmethod
    def save_contour_slice_from_sdf_grid_plt(
        sdf_grid: torch.tensor, save_path: Path, slice_num: int = 161, info: str = ""
    ) -> torch.tensor:
        assert sdf_grid.shape[0] == sdf_grid.shape[1] == sdf_grid.shape[2]
        dim = sdf_grid.shape[0]
        y_lims = [0, dim]
        z_lims = [0, dim]
        y_axis = np.linspace(*y_lims, num=dim)
        z_axis = np.linspace(*z_lims, num=dim)
        x = slice_num

        fig, ax = plt.subplots(1, 1)
        sdf_slice = sdf_grid[x, :, :]
        ax.contourf(y_axis, z_axis, sdf_slice, levels=20)
        if info:
            ax.set_title(f"Contour Plot - x={x} ({info})")
        else:
            ax.set_title(f"Contour Plot - x={x}")
        ax.set_xlabel("y-axis")
        ax.set_ylabel("z-axis")
        plt.savefig(str(save_path))
        plt.close()

    def get_pc_from_fs_buffer_and_pred_bbox(
        self,
        fs_buffer: struct.RGBDFrameSampleBuffer,
        pred_bbox: obj_det.YoloPredictionBoundingBox,
        valid_depth_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        assert len(fs_buffer) == 1, f"Expecting single frame only"
        depth_img = fs_buffer.frames_coll.frames[0].depth_img
        depth_img_shape = depth_img.shape

        label = pred_bbox.label
        xylist = pred_bbox.bbox
        depth_sampling_mask = torch.zeros(*depth_img_shape).float().to(self.device)
        depth_sampling_mask[xylist[0] : xylist[1], xylist[2] : xylist[3]] = 1.0
        depth_sampling_mask *= valid_depth_mask
        sample_pts = fs_buffer.get_surface_points_from_depth_batch(depth_sampling_mask)
        pc = sample_pts.pc
        if pc.numel() == 0:
            return None

        pc_mask_sdf = self.sdf_stc_pe_filter(pc.reshape(-1, 3))
        pc_mask_sdf = pc_mask_sdf.reshape(pc.shape[0], pc.shape[1])

        is_whole_ray_above_sdf = torch.all(pc_mask_sdf, dim=-1)
        surf_pc = pc[is_whole_ray_above_sdf][:, 0]

        if surf_pc.numel() == 0:
            # YOLO probably predicted a false positive
            return None

        # pre-filter object pc, based on dimensions
        surf_pc = self._prefilter_obj_pc_using_prior_info(
            pc_input=surf_pc, obj_id=label
        )
        surf_pc = self.fps_fn(surf_pc, self.fps_samples_max)
        return surf_pc

    def assign_pc_to_pred_bbox(
        self,
        fs_buffer: struct.RGBDFrameSampleBuffer,
        pred_bboxes: List[obj_det.YoloPredictionBoundingBox],
    ) -> List[tuple]:
        assert len(fs_buffer) == 1, f"Expecting single frame only"
        depth_img = fs_buffer.frames_coll.frames[0].depth_img
        valid_depth_mask = torch.tensor(
            (depth_img > 0.0) * (depth_img <= self.depth_max),
            dtype=torch.float32,
            device=self.device,
        )

        pred_bbox_depth_avg_pairs = []
        for pred_bbox in pred_bboxes:
            xylist = pred_bbox.bbox
            # get depth values from image using bbox
            depth_in_bbox = depth_img[xylist[0] : xylist[1], xylist[2] : xylist[3]]
            depth_in_bbox = depth_in_bbox[depth_in_bbox > 0.0]

            avg_depth = depth_in_bbox.mean()
            pred_bbox_depth_avg_pairs.append((pred_bbox, avg_depth))

        # ascending sorting
        pred_bbox_depth_avg_pairs = sorted(
            pred_bbox_depth_avg_pairs, key=lambda _x: _x[1]
        )

        # assigning pc nearest-first
        pred_bbox_pc_pairs = []
        _pc_comp_thresh = 0.1
        for bbox_depth_pair in pred_bbox_depth_avg_pairs:
            pred_bbox = bbox_depth_pair[0]
            surf_pc = self.get_pc_from_fs_buffer_and_pred_bbox(
                fs_buffer=fs_buffer,
                pred_bbox=pred_bbox,
                valid_depth_mask=valid_depth_mask,
            )
            if surf_pc is None:
                continue

            if len(pred_bbox_pc_pairs) > 0:
                # compare with previous pcs
                for bbox_pc_pair in pred_bbox_pc_pairs:
                    if surf_pc.numel() == 0:
                        break
                    surf_pc_prev = bbox_pc_pair[1]
                    dists = surf_pc[:, None, :] - surf_pc_prev[None, :, :]
                    dists = dists.norm(dim=-1).min(dim=-1).values
                    surf_pc = surf_pc[
                        dists > _pc_comp_thresh
                    ]  # retain pc different from prev ones

            if surf_pc.numel() == 0:
                continue
            pred_bbox_pc_pairs.append((pred_bbox, surf_pc))

        return pred_bbox_pc_pairs

    def get_fs_buffer_from_frame_sample(
        self, frame_sample: struct.RGBDFrameSample
    ) -> struct.RGBDFrameSampleBuffer:
        fsb = struct.RGBDFrameSampleBuffer(
            buffer_size=1,
            cam_params=self.cam_params,
            dirs_C=self.dirs_C,
            depth_min=0.09,
        )
        fsb.add_frame(frame_sample=frame_sample)
        return fsb

    def get_yolo_predictions(
        self, fs_buffer: struct.RGBDFrameSampleBuffer
    ) -> YoloFramePredictions:
        frame_sample = fs_buffer.frames_coll.frames[0]
        frame_preds = self.yolo.infer_bboxes_from_img(img=frame_sample.rgb_img)[0]
        return frame_preds

    def set_frame_sample_for_inference_no_yolo(
        self,
        obj_det_preds: YoloFramePredictions,
        fs_buffer: struct.RGBDFrameSampleBuffer,
    ):
        """
        This replaces set_frame_sample_for_inference to avoid duplicates in main
        pipeline.
        """
        self.__frame_bboxes = obj_det_preds  # yolo predictions already done
        # assign pc points to yolo predictions
        pred_bbox_pc_pairs = self.assign_pc_to_pred_bbox(
            fs_buffer=fs_buffer, pred_bboxes=self.__frame_bboxes.preds
        )
        self.__frame_bboxes.preds = [
            _pair[0] for _pair in pred_bbox_pc_pairs
        ]  # re-assign

        self.__tforms_list = []
        # valid_pred_count = 0
        for bbox_pc_pair in pred_bbox_pc_pairs:
            pred_bbox, surf_pc = bbox_pc_pair
            label = pred_bbox.label
            inf_param = self.compute_tform_for_input_pc_via_sdf(
                pc_input=surf_pc, obj_id=label
            )
            self.__tforms_list.append(inf_param)

    def set_frame_sample_for_inference(self, frame_sample: struct.RGBDFrameSample):
        # memory should be updated before this step
        fsb = self.get_fs_buffer_from_frame_sample(frame_sample=frame_sample)
        self.__frame_bboxes = self.get_yolo_predictions(fs_buffer=fsb)  # single batch

        # assign pc points to yolo predictions
        pred_bbox_pc_pairs = self.assign_pc_to_pred_bbox(
            fs_buffer=fsb, pred_bboxes=self.__frame_bboxes.preds
        )
        self.__frame_bboxes.preds = [
            _pair[0] for _pair in pred_bbox_pc_pairs
        ]  # re-assign

        self.__tforms_list = []
        valid_pred_count = 0
        for bbox_pc_pair in pred_bbox_pc_pairs:
            pred_bbox, surf_pc = bbox_pc_pair
            label = pred_bbox.label
            inf_param = self.compute_tform_for_input_pc_via_sdf(
                pc_input=surf_pc, obj_id=label
            )
            self.__tforms_list.append(inf_param)
            if inf_param is not None:
                valid_pred_count += 1
        print(f"[YOLO] only using {valid_pred_count} detections")

    # @torch.no_grad()
    def infer_sdf_at_points_with_preset_frame_sample(self, pt3ds: torch.Tensor):
        # sdf for scene
        sdf_vals = torch_utils.chunked_map(map=self.sdf_stc_pe, pt3ds=pt3ds)

        # do per obj sdf prediction and compose with scene sdf
        for idx, pred_bbox in enumerate(self.__frame_bboxes.preds):
            inf_param = self.__tforms_list[idx]
            label = pred_bbox.label
            if inf_param is None:
                # ignoring objects for which sufficient PC was not visible
                continue
            pt3ds_this = math_utils.apply_transform_to_pt3ds(
                pt3ds=pt3ds, tform=inf_param.tform
            )

            sdf_obj_values = torch_utils.chunked_map(
                map=self.sdf_obj.decode_sdf,
                pt3ds=pt3ds_this,
                latent_vector=self.sdf_obj_latvecs[label],
            )

            if sdf_vals is None:
                sdf_vals = sdf_obj_values * inf_param.scale
            else:
                sdf_vals = torch.minimum(sdf_vals, sdf_obj_values * inf_param.scale)

        # compose predictions for objects in memory
        for idx, obs_info in enumerate(self.memory.obs_list):
            tform = obs_info.inf_param.tform
            label = obs_info.label
            pt3ds_this = math_utils.apply_transform_to_pt3ds(pt3ds=pt3ds, tform=tform)
            sdf_obj_values = torch_utils.chunked_map(
                map=self.sdf_obj.decode_sdf,
                pt3ds=pt3ds_this,
                latent_vector=self.sdf_obj_latvecs[label],
            )

            if sdf_vals is None:
                sdf_vals = sdf_obj_values * obs_info.inf_param.scale
            else:
                sdf_vals = torch.minimum(
                    sdf_vals, sdf_obj_values * obs_info.inf_param.scale
                )

        return sdf_vals

    def scene_full_sdf_for_current_rgbd_frame_sample(
        self, frame_sample: Optional[struct.RGBDFrameSample] = None, grid_dim: int = 200
    ) -> torch.Tensor:
        # optimized version of scene_mesh_reconstruct()
        grid_pc = tform.make_3D_grid(
            grid_range=self.vis.grid_range,
            dim=grid_dim,
            device=self.device,
            transform=self.vis.bounds_transform_torch,
            scale=self.vis.scene_scale_torch,
        )
        if frame_sample is not None:
            self.set_frame_sample_for_inference(frame_sample=frame_sample)
            # frame_sample = None will assume it has already been set.
        sdf_values = self.infer_sdf_at_points_with_preset_frame_sample(
            pt3ds=grid_pc.view(-1, 3)
        )
        sdf_values = sdf_values.view(grid_dim, grid_dim, grid_dim)
        return sdf_values

    @torch.no_grad()
    def scene_full_sdf_for_map(
        self,
        sdf_map: Callable,
        grid_dim: int = 200,
        grid_tform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # optimized version of scene_mesh_reconstruct()
        grid_pc = tform.make_3D_grid(
            grid_range=self.vis.grid_range,
            dim=grid_dim,
            device=self.device,
            transform=self.vis.bounds_transform_torch,
            scale=self.vis.scene_scale_torch,
        )
        sdf_values = torch_utils.chunked_map(
            map=sdf_map, pt3ds=grid_pc.view(-1, 3), chunk_size=1000
        )
        sdf_values = sdf_values.view(grid_dim, grid_dim, grid_dim)
        return sdf_values


if __name__ == "__main__":
    config_dict = cmn.parse_yaml_file(
        filepath=Path("configs/inference_single_view.yaml")
    )
    inf_class = InferenceSingleViewAugSDF(config_dict=config_dict)
