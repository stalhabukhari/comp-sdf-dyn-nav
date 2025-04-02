from typing import Optional

import torch


def get_pc_bounds(pc: torch.Tensor) -> torch.Tensor:
    return torch.stack([pc.min(dim=0).values, pc.max(dim=0).values], dim=0)


def align_bbox_centers(
    pc_align_np: torch.Tensor, pc_base_np_bounds: torch.Tensor
) -> Optional[torch.Tensor]:
    # use pre-calculated bounds for pc_base_np to avoid extra op
    # added check for too-small-pc here for efficiency (returns None if too small)
    torch_device = pc_align_np.device
    bbox_bounds_align = get_pc_bounds(pc=pc_align_np)
    # check if pc_align is too small for accurate measurements:
    area_align = (bbox_bounds_align[1] - bbox_bounds_align[0]).prod()
    area_base = (pc_base_np_bounds[1] - pc_base_np_bounds[0]).prod()
    # assert area_align > 0 and area_base > 0
    if area_align <= 0 or area_base <= 0:
        return None
    if area_align / area_base < 0.05:  # less than 5%
        return None
    bbox_center_align = bbox_bounds_align.mean(dim=0)
    bbox_center_base = pc_base_np_bounds.mean(dim=0)
    offset = bbox_center_align - bbox_center_base
    tform = torch.eye(4, dtype=torch.float32, device=torch_device)
    tform[:3, 3] = -offset
    return tform
