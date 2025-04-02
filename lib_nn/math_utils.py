import torch


def apply_transform_to_pt3ds(pt3ds: torch.Tensor, tform: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(len(pt3ds), 1, dtype=torch.float32, device=pt3ds.device)
    pt4ds = torch.cat((pt3ds, ones), dim=-1)
    pt4ds = (pt4ds[:, None] * tform[None, ...]).sum(dim=-1)
    return pt4ds[:, :3]


def apply_batched_transform_to_pt3ds(
    pt3ds: torch.Tensor, tforms: torch.Tensor
) -> torch.Tensor:
    """
    :param pt3ds: Nx3 -: ?xNx?x3
    :param tforms: Mx4x4 -> Mx?x4x4
    :return: MxNx3
    """
    ones = torch.ones(len(pt3ds), 1, dtype=torch.float32, device=pt3ds.device)
    pt4ds = torch.cat((pt3ds, ones), dim=-1)
    pt4ds = (pt4ds[None, :, None, :] * tforms[:, None, ...]).sum(dim=-1)
    return pt4ds[..., :3]


def skew_symmetric_mat_from_vec(vec: torch.Tensor) -> torch.Tensor:
    skew_mat = torch.zeros(3, 3, dtype=torch.float32, device=vec.device)
    skew_mat[0, 1] += -vec[2]
    skew_mat[0, 2] += vec[1]
    skew_mat[1, 0] += vec[2]
    skew_mat[1, 2] += -vec[0]
    skew_mat[2, 0] += -vec[1]
    skew_mat[2, 1] += vec[0]
    return skew_mat


def skew_symmetric_mat_from_vec_batched(vec: torch.Tensor) -> torch.Tensor:
    # TODO: revisit along with optimize_rotz_tran_batched()
    N = vec.shape[0]
    skew_mat = torch.zeros(N, 3, 3, dtype=torch.float32, device=vec.device)
    skew_mat[:, 0, 1] += -vec[:, 2]
    skew_mat[:, 0, 2] += vec[:, 1]
    skew_mat[:, 1, 0] += vec[:, 2]
    skew_mat[:, 1, 2] += -vec[:, 0]
    skew_mat[:, 2, 0] += -vec[:, 1]
    skew_mat[:, 2, 1] += vec[:, 0]
    return skew_mat


def rot_mat_from_rot_vec_and_angle(
    rot_vec: torch.Tensor, rot_ang: torch.Tensor
) -> torch.Tensor:
    skew_mat = skew_symmetric_mat_from_vec(vec=rot_vec)
    rot_vec_norm = rot_vec.norm()
    rot_mat = torch.eye(3, dtype=torch.float32, device=rot_vec.device)
    rot_mat = (
        rot_mat
        + skew_mat / rot_vec_norm * torch.sin(rot_vec_norm * rot_ang)
        + (skew_mat @ skew_mat)
        / rot_vec_norm.square()
        * (1 - torch.cos(rot_vec_norm * rot_ang))
    )
    return rot_mat


def rot_mat_from_rot_vec_and_angle_batched(
    rot_vec: torch.Tensor, rot_ang: torch.Tensor
) -> torch.Tensor:
    """
    :param rot_vec: Nx3 
    :param rot_ang: Nx1
    :return: rot_mat: Nx3x3
    """ ""
    N = rot_vec.shape[0]
    skew_mat = skew_symmetric_mat_from_vec_batched(vec=rot_vec)
    rot_vec_norm = rot_vec.norm(dim=-1, keepdim=True)
    rot_mat = (
        torch.eye(3, dtype=torch.float32, device=rot_vec.device)
        .reshape(1, 3, 3)
        .repeat(N, 1, 1)
    )
    rot_mat = (
        rot_mat
        + skew_mat / rot_vec_norm * torch.sin(rot_vec_norm * rot_ang)
        + skew_mat.bmm(skew_mat)
        / rot_vec_norm.square()
        * (1 - torch.cos(rot_vec_norm * rot_ang))
    )
    return rot_mat
