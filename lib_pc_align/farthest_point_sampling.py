from typing import Callable, Union, Tuple

import numpy as np
import torch
from rich import print

try:
    from pointnet2_ops import pointnet2_utils

    FPS = "CUDA"
except Exception as _e_:
    print(f"[FPS] Reverting to Open3D due to exception: {_e_}")
    import open3d as o3d

    FPS = "O3D"


def fps_cuda(
    pc: torch.Tensor, num_pts_max: int, index: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if num_pts_max >= pc.shape[0]:
        return pc
    pc = pc.unsqueeze(0).contiguous()
    idxs = pointnet2_utils.furthest_point_sample(pc, num_pts_max)
    pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), idxs)
    pc = pc.transpose(1, 2).squeeze(0)
    if index:
        return pc, idxs
    return pc


def fps_o3d(
    pc: torch.Tensor, num_pts_max: int, index: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if num_pts_max >= pc.shape[0]:
        return pc
    torch_device = pc.device
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
    pcd2 = pcd1.farthest_point_down_sample(num_samples=num_pts_max)
    pc_input_subsampled = torch.tensor(
        np.asarray(pcd2.points), dtype=torch.float32, device=torch_device
    )
    # get indices by pt2pt comparison: (may need to do chunked)
    diffs = (pc[None, :] - pc_input_subsampled[:, None]).norm(dim=-1)
    idxs = diffs.argmin(dim=1)
    if index:
        return pc_input_subsampled, idxs
    return pc_input_subsampled


def get_fps_fn() -> Callable:
    if FPS == "CUDA":
        return fps_cuda
    return fps_o3d
