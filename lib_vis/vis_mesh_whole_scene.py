from pathlib import Path
from typing import Dict, Callable, Union

import numpy as np
import torch
import trimesh
from rich import print

from data import transform as tform
from lib_nn import sdf
from lib_nn import torch_utils
from data import mesh_utils
from lib_vis import vis_utils


class Visualizer:
    def __init__(
        self,
        config_dict: Dict,
        scene_mesh_file: Path,
        scene_data_file: Path = None,
        load_sdf_ckpt: bool = False,
    ):
        assert torch.cuda.is_available()
        # assert scene_mesh_file.exists()
        self.config_dict = config_dict
        self.device = torch.device("cuda")
        if scene_data_file is not None:
            self._set_scene_params_from_npy_file(scene_data_file)
        else:
            self._set_scene_params_from_scene_mesh(raw_scene_mesh_path=scene_mesh_file)
        if load_sdf_ckpt:
            self._load_sdf_model()

    def _set_scene_params_from_scene_mesh(self, raw_scene_mesh_path: Path):
        assert raw_scene_mesh_path.exists()
        raw_scene_mesh = trimesh.exchange.load.load(
            str(raw_scene_mesh_path), process=False
        )
        scene_props = mesh_utils.get_scene_props_from_scene_mesh(
            scene_mesh=raw_scene_mesh
        )

        self.bounds_transform = np.linalg.inv(scene_props.bounds_transform_inv)
        self.bounds_transform_torch = (
            torch.from_numpy(self.bounds_transform).float().to(self.device)
        )
        self.bounds_transform_inv_torch = (
            torch.from_numpy(scene_props.bounds_transform_inv).float().to(self.device)
        )
        self.scene_center = scene_props.scene_center
        bounds_extent = scene_props.bounds_extent

        self.grid_range = [-1.0, 1.0]
        range_dist = self.grid_range[1] - self.grid_range[0]
        self.scene_scale = bounds_extent / (range_dist * 0.9)
        # TODO: remove 0.9 (see if this impacts filtering out background pc)
        self.scene_scale_torch = (
            torch.from_numpy(self.scene_scale).float().to(self.device)
        )

    def _set_scene_params_from_npy_file(self, npy_filepath: Path):
        assert npy_filepath.exists()
        T_extent_to_scene = np.load(str(npy_filepath / "scene_T_extent_to_scene.npy"))
        bounds_extents = np.load(str(npy_filepath / "scene_bounds_extents.npy"))
        scene_center = np.load(str(npy_filepath / "scene_scene_center.npy"))
        scene_props = mesh_utils.get_scene_props_from_npyfiles(
            T_extent_to_scene, bounds_extents, scene_center
        )

        self.bounds_transform = np.linalg.inv(scene_props.bounds_transform_inv)
        self.bounds_transform_torch = (
            torch.from_numpy(self.bounds_transform).float().to(self.device)
        )
        self.bounds_transform_inv_torch = (
            torch.from_numpy(scene_props.bounds_transform_inv).float().to(self.device)
        )
        self.scene_center = scene_props.scene_center
        bounds_extent = scene_props.bounds_extent

        self.grid_range = [-1.0, 1.0]
        range_dist = self.grid_range[1] - self.grid_range[0]
        self.scene_scale = bounds_extent / (range_dist * 0.9)
        # TODO: remove 0.9 (see if this impacts filtering out background pc)
        self.scene_scale_torch = (
            torch.from_numpy(self.scene_scale).float().to(self.device)
        )

    def _load_sdf_model(self):
        model_config = self.config_dict["model"]
        model_scale_output = model_config["scale_output"]
        ckpt_dir = Path(model_config["ckpt_dir"])

        pos_embed_config = model_config["positional_embedding"]
        pos_embed_num_embeds = pos_embed_config["num_embed_fns"]
        pos_embed_scale_input = pos_embed_config["scale_input"]

        sdf_stc_config = model_config["sdf_static"]
        sdf_stc_hidden_feat = sdf_stc_config["hidden_feature_size"]
        sdf_stc_hidden_blk = sdf_stc_config["hidden_layers_block"]
        sdf_stc_ckpt = ckpt_dir / sdf_stc_config["ckpt"]

        self.pose_enc = sdf.PositionalEncoding(
            min_deg=0,
            max_deg=pos_embed_num_embeds,
            scale=pos_embed_scale_input,
            transform=self.bounds_transform_inv_torch,
        ).to(self.device)
        self.pose_enc.eval()

        self.sdf_stc = sdf.SDFMap(
            pos_enc_embedding_size=self.pose_enc.embedding_size,
            hidden_size=sdf_stc_hidden_feat,
            hidden_layers_block=sdf_stc_hidden_blk,
            scale_output=model_scale_output,
        ).to(self.device)
        torch_utils.load_checkpoint(model=self.sdf_stc, ckpt_filepath=sdf_stc_ckpt)
        self.sdf_stc.eval()

        def _sdf_map(x: torch.tensor) -> torch.tensor:
            x_pe = self.pose_enc(x)
            sdf_pred = self.sdf_stc(x_pe)
            return sdf_pred

        self.sdf_map = _sdf_map

    def set_sdf_map(self, sdf_map: Union[torch.nn.Module, Callable]):
        self.sdf_map = sdf_map

    def get_sdf_queried_on_grid(
        self, x_grid: torch.tensor, chunk_size: int, grid_dim: int
    ) -> torch.tensor:
        with torch.set_grad_enabled(False):
            sdf_pred = torch_utils.chunks(x_grid, chunk_size, self.sdf_map)
            dim = grid_dim
            sdf_pred = sdf_pred.view(dim, dim, dim)

        return sdf_pred

    def draw_mesh(
        self, grid_dim: int = 200, chunk_size: int = 100000
    ) -> trimesh.Trimesh:
        grid_pc = tform.make_3D_grid(
            self.grid_range,
            grid_dim,
            self.device,
            transform=self.bounds_transform_torch,
            scale=self.scene_scale_torch,
        )
        grid_pc = grid_pc.view(-1, 3).to(self.device)

        sdf_pred = self.get_sdf_queried_on_grid(
            x_grid=grid_pc, chunk_size=chunk_size, grid_dim=grid_dim
        )
        sdf_mesh = vis_utils.draw_mesh(
            sdf_pred, self.scene_scale, self.bounds_transform
        )
        return sdf_mesh

    @staticmethod
    def save_mesh(mesh: trimesh.Trimesh, save_filepath: Path):
        assert save_filepath.suffix == ".obj"
        mesh.export(str(save_filepath))
        print(f"Saved mesh: {save_filepath}")
