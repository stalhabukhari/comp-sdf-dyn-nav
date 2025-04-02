from typing import Tuple, Optional

import torch

from data import transform as tform


def sample_pixels_torch(
    n_rays: int,
    n_frames: int,
    h: int,
    w: int,
    device: torch.device,
    sampling_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    if sampling_mask is not None:
        assert n_rays < 0
        n_rays = (sampling_mask > 0).sum().to(torch.int)
        indices_2d = torch.nonzero(sampling_mask, as_tuple=True)
        indices_h = indices_2d[0].to(device)
        indices_w = indices_2d[1].to(device)
        indices_b = torch.arange(n_frames, device=device)
        indices_b = indices_b.repeat_interleave(n_rays)
    else:
        if n_rays < 0:
            n_rays = h * w
            indices_2d = torch.cartesian_prod(
                torch.arange(0, h, device=device), torch.arange(0, w, device=device)
            )
            indices_h = indices_2d[:, 0]
            indices_w = indices_2d[:, 1]
            indices_b = torch.arange(n_frames, device=device)
            indices_b = indices_b.repeat_interleave(n_rays)

        else:
            total_rays = n_rays * n_frames
            indices_h = torch.randint(0, h, (total_rays,), device=device)
            indices_w = torch.randint(0, w, (total_rays,), device=device)

            indices_b = torch.arange(n_frames, device=device)
            indices_b = indices_b.repeat_interleave(n_rays)

    return indices_b, indices_h, indices_w


def get_batch_data_torch(
    depth_batch: torch.tensor,
    T_WC_batch: torch.tensor,
    dirs_C: torch.tensor,
    indices_b: torch.tensor,
    indices_h: torch.tensor,
    indices_w: torch.tensor,
    norm_batch: Optional[torch.tensor] = None,
    get_masks: bool = False,
) -> Tuple:
    """
    Get depth, ray direction and pose for the sampled pixels.
    Only render where depth is valid.
    """
    depth_sample = depth_batch[indices_b, indices_h, indices_w].view(-1)
    mask_valid_depth = depth_sample != 0

    norm_sample = None
    if norm_batch is not None:
        norm_sample = norm_batch[indices_b, indices_h, indices_w, :].view(-1, 3)
        mask_invalid_norm = torch.isnan(norm_sample[..., 0])
        mask_valid_depth = torch.logical_and(mask_valid_depth, ~mask_invalid_norm)
        norm_sample = norm_sample[mask_valid_depth]

    depth_sample = depth_sample[mask_valid_depth]

    indices_b = indices_b[mask_valid_depth]
    indices_h = indices_h[mask_valid_depth]
    indices_w = indices_w[mask_valid_depth]

    T_WC_sample = T_WC_batch[indices_b]
    dirs_C_sample = dirs_C[0, indices_h, indices_w, :].view(-1, 3)

    masks = None
    if get_masks:
        masks = torch.zeros(depth_batch.shape, device=depth_batch.device)
        masks[indices_b, indices_h, indices_w] = 1

    return (
        dirs_C_sample,
        depth_sample,
        norm_sample,
        T_WC_sample,
        masks,
        indices_b,
        indices_h,
        indices_w,
    )


def stratified_sample(
    min_depth,
    max_depth,
    n_rays,
    device,
    n_stratified_samples,
    bin_length=None,
):
    """
    Random samples between min and max depth
    One sample from within each bin.

    If n_stratified_samples is passed then use fixed number of bins,
    else if bin_length is passed use fixed bin size.
    """
    if n_stratified_samples is not None:  # fixed number of bins
        n_bins = n_stratified_samples
        if isinstance(max_depth, torch.Tensor):
            sample_range = (max_depth - min_depth)[:, None]
            bin_limits = torch.linspace(0, 1, n_bins + 1, device=device)[None, :]
            bin_limits = bin_limits.repeat(n_rays, 1) * sample_range
            if isinstance(min_depth, torch.Tensor):
                bin_limits = bin_limits + min_depth[:, None]
            else:
                bin_limits = bin_limits + min_depth
            bin_length = sample_range / (n_bins)
        else:
            raise Exception("Unexpected behavior in `stratified_sample`")

    elif bin_length is not None:  # fixed size of bins
        bin_limits = torch.arange(
            min_depth,
            max_depth,
            bin_length,
            device=device,
        )[None, :]
        n_bins = bin_limits.size(1) - 1

    else:
        raise Exception(
            "Either n_stratified_samples or bin_length should be provided. Both can't be None"
        )

    increments = torch.rand(n_rays, n_bins, device=device) * bin_length
    # increments = 0.5 * torch.ones(n_rays, n_bins, device=device) * bin_length
    lower_limits = bin_limits[..., :-1]
    z_vals = lower_limits + increments

    return z_vals


def sample_along_rays_till_depth(
    min_depth,
    max_depth,
    n_rays,
    device,
    n_stratified_samples,
    bin_length=None,
):
    """
    Random samples between min and max depth
    One sample from within each bin.

    If n_stratified_samples is passed then use fixed number of bins,
    else if bin_length is passed use fixed bin size.
    """
    if n_stratified_samples is not None:  # fixed number of bins
        n_bins = n_stratified_samples
        if isinstance(max_depth, torch.Tensor):
            sample_range = (max_depth - min_depth)[:, None]
            bin_limits = torch.linspace(0, 1, n_bins + 1, device=device)[None, :]
            bin_limits = bin_limits.repeat(n_rays, 1) * sample_range
            if isinstance(min_depth, torch.Tensor):
                bin_limits = bin_limits + min_depth[:, None]
            else:
                bin_limits = bin_limits + min_depth
            bin_length = sample_range / (n_bins)
        else:
            raise Exception("Unexpected behavior in `stratified_sample`")

    elif bin_length is not None:  # fixed size of bins
        bin_limits = torch.arange(
            min_depth,
            max_depth,
            bin_length,
            device=device,
        )[None, :]
        n_bins = bin_limits.size(1) - 1

    else:
        raise Exception(
            "Either n_stratified_samples or bin_length should be provided. Both can't be None"
        )

    increments = torch.rand(n_rays, n_bins, device=device) * bin_length
    # increments = 0.5 * torch.ones(n_rays, n_bins, device=device) * bin_length
    lower_limits = bin_limits[..., :-1]
    z_vals = lower_limits + increments

    return z_vals


def sample_along_rays(
    T_WC: torch.tensor,
    min_depth: torch.tensor,
    max_depth: torch.tensor,
    n_stratified_samples: int,
    n_surf_samples: int,
    dirs_C: torch.tensor,
    gt_depth: Optional[torch.tensor] = None,
    grad=False,
):
    with torch.set_grad_enabled(grad):
        # rays in world coordinate
        origins, dirs_W = tform.origin_dirs_W(T_WC, dirs_C)

        origins = origins.view(-1, 3)
        dirs_W = dirs_W.view(-1, 3)
        n_rays = dirs_W.shape[0]

        # stratified sampling along rays # [total_n_rays, n_stratified_samples]
        z_vals = stratified_sample(
            min_depth,
            max_depth,
            n_rays,
            T_WC.device,
            n_stratified_samples,
            bin_length=None,
        )

        # if gt_depth is given, first sample at surface then around surface
        if gt_depth is not None and n_surf_samples > 0:
            surface_z_vals = gt_depth[:, None]
            to_cat = [surface_z_vals]
            if n_surf_samples > 1:
                offsets = torch.normal(
                    torch.zeros(gt_depth.shape[0], n_surf_samples - 1), 0.1
                ).to(z_vals.device)
                # n_surf_samples - 1, since 1 sample is always the depth sample.
                # assumption: changes in surface topology are slow.
                # if we reduce the std (0.1), would it be beneficial?
                near_surf_z_vals = gt_depth[:, None] + offsets
                if not isinstance(min_depth, torch.Tensor):
                    # TODO: if min_depth is np.ndarray, I get a type inference
                    #  error from torch.full()
                    min_depth = torch.full(near_surf_z_vals.shape, min_depth).to(
                        z_vals.device
                    )[
                        ..., 0
                    ]  # taking only the first value, could have just vectorized min_depth
                near_surf_z_vals = torch.clamp(
                    near_surf_z_vals, min_depth[:, None], max_depth[:, None]
                )
                to_cat.append(near_surf_z_vals)
            to_cat.append(z_vals)
            z_vals = torch.cat(to_cat, dim=1)

        # point cloud of 3d sample locations
        pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
        pc_loc = dirs_C[:, None, :] * z_vals[:, :, None]  # points in camera's CRS
    return pc, pc_loc, z_vals


def sample_along_rays_uniformly(
    T_WC: torch.tensor,
    n_stratified_samples: Optional[int],
    dirs_C: torch.tensor,
    gt_depth: Optional[torch.tensor] = None,
    grad=False,
) -> Tuple:
    with torch.set_grad_enabled(grad):
        # rays in world coordinate
        origins, dirs_W = tform.origin_dirs_W(T_WC, dirs_C)

        origins = origins.view(-1, 3)
        dirs_W = dirs_W.view(-1, 3)
        n_rays = dirs_W.shape[0]

        # stratified sampling along rays # [total_n_rays, n_stratified_samples]
        z_vals = sample_along_rays_till_depth(
            0.09, gt_depth, n_rays, T_WC.device, n_stratified_samples, bin_length=0.05
        )
        surface_z_vals = gt_depth[:, None]
        z_vals = torch.cat((surface_z_vals, z_vals), dim=1)
        # point cloud of 3d sample locations
        pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
    return pc, z_vals


def depth_image_to_point_cloud(
    T_WC: torch.tensor, dirs_C: torch.tensor, gt_depth: torch.tensor, grad=False
):
    """simplified version of sample_along_rays"""
    with torch.set_grad_enabled(grad):
        # rays in world coordinate
        origins, dirs_W = tform.origin_dirs_W(T_WC, dirs_C)
        origins = origins.view(-1, 3)
        dirs_W = dirs_W.view(-1, 3)
        # if gt_depth is given, first sample at surface then around surface
        surface_z_vals = gt_depth[:, None]
        # point cloud of 3d sample locations
        pc = origins[:, None, :] + (dirs_W[:, None, :] * surface_z_vals[:, :, None])
        pc_loc = (
            dirs_C[:, None, :] * surface_z_vals[:, :, None]
        )  # points in camera's CRS
    return pc, pc_loc, surface_z_vals
