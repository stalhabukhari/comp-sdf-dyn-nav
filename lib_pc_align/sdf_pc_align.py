from typing import Callable, Union, List, Tuple

import torch
from torch import nn

from lib_nn import math_utils
from data import common as cmn

_EPS = 0.01


class PcAlignSdfGradOpt:
    def __init__(
        self,
        iterations: int = 30,
        lr: float = 1e-3,
        lr_decay: float = 0.75,
        device: torch.device = torch.device("cuda"),
    ):
        self.iterations = iterations
        self.lr = lr
        self.lr_decay = lr_decay
        self.device = device

    def optimize_rotz_tran(
        self,
        pt3ds: torch.Tensor,
        sdf_map: Union[Callable, nn.Module],
        rot_ang_init: float,
    ) -> torch.Tensor:
        # my rot_init was np.pi / 4 or (np.pi / 4 + np.pi)
        rot_vec = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        )  # fixed along z-axis
        rot_ang = torch.tensor([rot_ang_init], dtype=torch.float32, device=self.device)
        rot_ang.requires_grad = True
        translate = torch.tensor(
            [_EPS] * 3, dtype=torch.float32, device=self.device
        ).requires_grad_(True)
        opt = torch.optim.Adam(
            [
                {"params": rot_ang, "lr": self.lr},
                {"params": translate, "lr": self.lr * 0.01},
            ]
        )

        lower_sdf_margin = -0.2
        upper_sdf_margin = 0.01
        for it in range(
            self.iterations
        ):  # tested with 130 iterations with lr_init=0.99
            if it % 50 == 0:
                lower_sdf_margin = min(lower_sdf_margin + 0.02, -0.05)
                upper_sdf_margin = max(upper_sdf_margin - 0.05, 0)
            # infer transform
            rot_mat = math_utils.rot_mat_from_rot_vec_and_angle(
                rot_vec=rot_vec, rot_ang=rot_ang
            )
            tform_init = torch.eye(4, dtype=torch.float32, device=self.device)
            tform_init[:3, :3] = rot_mat
            tform_init[:3, 3] = translate
            # data
            pts_tformed = math_utils.apply_transform_to_pt3ds(
                pt3ds=pt3ds, tform=tform_init
            )
            # predict sdf
            sdf_out = sdf_map(queries=pts_tformed)
            m1 = (sdf_out > upper_sdf_margin).float()
            m2 = (sdf_out < lower_sdf_margin).float()
            loss = torch.exp(sdf_out.square()) * (m1 + m2)
            loss = loss.mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            for pg in opt.param_groups:
                pg["lr"] *= self.lr_decay

        with torch.no_grad():
            rot_mat_out = math_utils.rot_mat_from_rot_vec_and_angle(
                rot_vec=rot_vec, rot_ang=rot_ang
            )
        tform_out = torch.eye(4, dtype=torch.float32, device=self.device)
        tform_out[:3, :3] = rot_mat_out
        tform_out[:3, 3] = translate.detach()
        return tform_out

    def optimize_rotz_tran_with_loss(
        self,
        pt3ds: torch.Tensor,
        sdf_map: Union[Callable, nn.Module],
        rot_ang_init: float,
    ) -> Tuple:
        # my rot_init was np.pi / 4 or (np.pi / 4 + np.pi)
        rot_vec = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        )  # fixed along z-axis
        rot_ang = torch.tensor([rot_ang_init], dtype=torch.float32, device=self.device)
        rot_ang.requires_grad = True
        translate = torch.tensor(
            [_EPS] * 3, dtype=torch.float32, device=self.device
        ).requires_grad_(True)
        opt = torch.optim.Adam(
            [
                {"params": rot_ang, "lr": self.lr},
                {"params": translate, "lr": self.lr * 0.01},
            ]
        )

        lower_sdf_margin = -0.2
        upper_sdf_margin = 0.01
        for it in range(
            self.iterations
        ):  # tested with 130 iterations with lr_init=0.99
            if it % 50 == 0:
                lower_sdf_margin = min(lower_sdf_margin + 0.02, -0.05)
                upper_sdf_margin = max(upper_sdf_margin - 0.05, 0)
            # infer transform
            rot_mat = math_utils.rot_mat_from_rot_vec_and_angle(
                rot_vec=rot_vec, rot_ang=rot_ang
            )
            tform_init = torch.eye(4, dtype=torch.float32, device=self.device)
            tform_init[:3, :3] = rot_mat
            tform_init[:3, 3] = translate
            # data
            pts_tformed = math_utils.apply_transform_to_pt3ds(
                pt3ds=pt3ds, tform=tform_init
            )
            # predict sdf
            sdf_out = sdf_map(queries=pts_tformed)
            # loss
            m1 = (sdf_out > upper_sdf_margin).float()
            m2 = (sdf_out < lower_sdf_margin).float()
            loss = torch.exp(sdf_out.square()) * (m1 + m2)
            loss = loss.mean()
            # backprop
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            # lr decay
            for pg in opt.param_groups:
                pg["lr"] *= self.lr_decay

        with torch.no_grad():
            rot_mat_out = math_utils.rot_mat_from_rot_vec_and_angle(
                rot_vec=rot_vec, rot_ang=rot_ang
            )
        tform_out = torch.eye(4, dtype=torch.float32, device=self.device)
        tform_out[:3, :3] = rot_mat_out
        tform_out[:3, 3] = translate.detach()

        # final forward pass to calculate loss
        pts_tformed = math_utils.apply_transform_to_pt3ds(pt3ds=pt3ds, tform=tform_out)
        loss = torch.exp(sdf_map(queries=pts_tformed).square()).mean()

        return tform_out, loss.detach().cpu().item()

    def optimize_rotz_tran_batched(
        self,
        pt3ds: torch.Tensor,
        sdf_map: Union[Callable, nn.Module],
        rot_ang_inits: List[float],
    ) -> torch.Tensor:
        N = len(rot_ang_inits)
        M = pt3ds.shape[0]
        rot_ang = torch.tensor(
            rot_ang_inits, dtype=torch.float32, device=self.device
        ).reshape(N, 1)
        translate = torch.tensor(
            [[_EPS] * 3] * N, dtype=torch.float32, device=self.device
        ).reshape(N, 3)
        rot_ang.requires_grad = True
        translate.requires_grad = True
        opt = torch.optim.Adam(
            [
                {"params": rot_ang, "lr": self.lr},
                {"params": translate, "lr": self.lr * 0.01},
            ]
        )

        # fixed along z-axis
        rot_vec = (
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
            .reshape(1, 3)
            .repeat(N, 1)
        )

        lower_sdf_margin = -0.2
        upper_sdf_margin = 0.01
        loss_k = None
        for it in range(
            self.iterations
        ):  # tested with 130 iterations with lr_init=0.99
            if it % 50 == 0:
                lower_sdf_margin = min(lower_sdf_margin + 0.02, -0.05)
                upper_sdf_margin = max(upper_sdf_margin - 0.05, 0)
            # infer transform
            rot_mats = math_utils.rot_mat_from_rot_vec_and_angle_batched(
                rot_vec=rot_vec, rot_ang=rot_ang
            )
            tforms_init = (
                torch.eye(4, dtype=torch.float32, device=self.device)
                .reshape(1, 4, 4)
                .repeat(N, 1, 1)
            )
            tforms_init[:, :3, :3] = rot_mats
            tforms_init[:, :3, 3] = translate
            # data
            pts_tformed_batch = math_utils.apply_batched_transform_to_pt3ds(
                pt3ds=pt3ds, tforms=tforms_init
            )
            # predict sdf
            sdf_out = sdf_map(queries=pts_tformed_batch.reshape(-1, 3)).reshape(N, M)
            m1 = (sdf_out > upper_sdf_margin).float()
            m2 = (sdf_out < lower_sdf_margin).float()
            loss = torch.exp(sdf_out.square()) * (m1 + m2)
            loss_k = loss.mean(dim=-1)
            loss = loss_k.mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            for pg in opt.param_groups:
                pg["lr"] *= self.lr_decay

        with torch.no_grad():
            min_idx = loss_k.argmin()

            # get tform
            rot_mat_out = math_utils.rot_mat_from_rot_vec_and_angle(
                rot_vec=rot_vec[min_idx], rot_ang=rot_ang[min_idx]
            )
            tform_out = torch.eye(4, dtype=torch.float32, device=self.device)
            tform_out[:3, :3] = rot_mat_out
            tform_out[:3, 3] = translate[min_idx].detach()

        return tform_out

    @torch.no_grad()
    def optimize_rot_only_discrete(
        self,
        pt3ds: torch.Tensor,
        sdf_map: Union[Callable, nn.Module],
        ang_res: float = 2.0 * torch.pi / 360.0,
    ) -> torch.Tensor:
        def _z_rot_ang_2_tform_mat(_ang: torch.Tensor) -> torch.Tensor:
            _tform = (
                torch.tensor(
                    [
                        [torch.cos(_ang), -torch.sin(_ang), 0.0, 0.0],
                        [torch.sin(_ang), torch.cos(_ang), 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                .float()
                .to(self.device)
            )
            return _tform

        # rotation transforms
        angles_all = torch.arange(0, 2 * torch.pi, ang_res).to(self.device)
        rot_tforms = []
        for ang in angles_all:
            # can remove for loop via repmat
            tform = _z_rot_ang_2_tform_mat(_ang=ang)
            rot_tforms.append(tform)
        rot_tforms = torch.stack(rot_tforms, dim=0)

        # subsample points
        indices = torch.randperm(pt3ds.shape[0])[: int(pt3ds.shape[0] / 20)]
        pt3ds = pt3ds[indices]

        M = rot_tforms.shape[0]
        N = pt3ds.shape[0]

        rot_tforms = rot_tforms[:, None]  # M x 1 x (4 x 4)
        pt3ds = torch.cat((pt3ds, torch.ones(N, 1).float().to(self.device)), dim=-1)
        pt3ds = pt3ds[None, :, None, :]  # 1 x N x 1 x 4
        pt3ds_tf = (pt3ds * rot_tforms).sum(dim=-1)  # M x N x 4

        assert tuple([*pt3ds_tf.shape]) == (M, N, 4)
        pt3ds_tf = pt3ds_tf[:, :, :3].reshape(-1, 3)
        # compute loss
        out = sdf_map(queries=pt3ds_tf)
        out = out.reshape(M, N).abs().sum(dim=-1)
        min_idx = out.argmin()
        ang = angles_all[min_idx]
        # convert to tform
        tform_out = _z_rot_ang_2_tform_mat(_ang=ang)
        return tform_out
