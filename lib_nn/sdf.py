from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print


def transform_3D_grid(grid_3d, transform=None, scale=None):
    # TODO: move to a separate file
    if scale is not None:
        grid_3d = grid_3d * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)

        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d


def scale_input(tensor, transform=None, scale=None):
    # TODO: move to a separate file
    if transform is not None:
        t_shape = tensor.shape
        tensor = transform_3D_grid(tensor.view(-1, 3), transform=transform)
        tensor = tensor.view(t_shape)

    if scale is not None:
        tensor = tensor * scale

    return tensor


def init_weights(m, init_fn=torch.nn.init.xavier_normal_):
    if isinstance(m, torch.nn.Linear):
        init_fn(m.weight)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, min_deg=0, max_deg=6, scale=0.1, transform=None):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.n_freqs = max_deg - min_deg + 1
        self.scale = scale
        self.transform = transform

        # TODO: revisit this 21x3 matrix
        self.dirs = (
            torch.tensor(
                [
                    0.8506508,
                    0,
                    0.5257311,
                    0.809017,
                    0.5,
                    0.309017,
                    0.5257311,
                    0.8506508,
                    0,
                    1,
                    0,
                    0,
                    0.809017,
                    0.5,
                    -0.309017,
                    0.8506508,
                    0,
                    -0.5257311,
                    0.309017,
                    0.809017,
                    -0.5,
                    0,
                    0.5257311,
                    -0.8506508,
                    0.5,
                    0.309017,
                    -0.809017,
                    0,
                    1,
                    0,
                    -0.5257311,
                    0.8506508,
                    0,
                    -0.309017,
                    0.809017,
                    -0.5,
                    0,
                    0.5257311,
                    0.8506508,
                    -0.309017,
                    0.809017,
                    0.5,
                    0.309017,
                    0.809017,
                    0.5,
                    0.5,
                    0.309017,
                    0.809017,
                    0.5,
                    -0.309017,
                    0.809017,
                    0,
                    0,
                    1,
                    -0.5,
                    0.309017,
                    0.809017,
                    -0.809017,
                    0.5,
                    0.309017,
                    -0.809017,
                    0.5,
                    -0.309017,
                ]
            )
            .reshape(-1, 3)
            .T
        )

        frequency_bands = 2.0 ** np.linspace(self.min_deg, self.max_deg, self.n_freqs)
        self.embedding_size = 2 * self.dirs.shape[1] * self.n_freqs + 3

        print(
            f"[Debug] Icosahedron embedding with periods:"
            f"{(2 * np.pi) / (frequency_bands * self.scale)}"
            f" -- embedding size: {self.embedding_size}"
        )

    def vis_embedding(self):
        x = torch.linspace(0, 5, 640)
        embd = x * self.scale
        if self.gauss_embed:
            frequency_bands = torch.norm(self.B_layer.weight, dim=1)
            frequency_bands = torch.sort(frequency_bands)[0]
        else:
            frequency_bands = 2.0 ** torch.linspace(
                self.min_deg, self.max_deg, self.n_freqs
            )

        embd = embd[..., None] * frequency_bands
        embd = torch.sin(embd)

        import matplotlib.pylab as plt

        plt.imshow(
            embd.T,
            cmap="hot",
            interpolation="nearest",
            aspect="auto",
            extent=[0, 5, 0, embd.shape[1]],
        )
        plt.colorbar()
        plt.xlabel("x values")
        plt.ylabel("embedings")
        plt.show()

    def forward(self, tensor: torch.tensor) -> torch.tensor:
        frequency_bands = 2.0 ** torch.linspace(
            self.min_deg,
            self.max_deg,
            self.n_freqs,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        tensor = scale_input(tensor, transform=self.transform, scale=self.scale)

        proj = torch.matmul(tensor, self.dirs.to(tensor.device))
        xb = torch.reshape(
            proj[..., None] * frequency_bands, list(proj.shape[:-1]) + [-1]
        )
        embedding = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
        embedding = torch.cat([tensor] + [embedding], dim=-1)

        return embedding


class FCBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.acti = nn.Softplus(beta=100)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc(x)
        x = self.acti(x)
        return x


class IsdfMap(nn.Module):
    def __init__(
        self,
        pos_enc_embedding_size: int,
        hidden_size: int = 256,
        hidden_layers_block: int = 1,
        scale_output: int = 1.0,
    ):
        super(IsdfMap, self).__init__()
        self.scale_output = scale_output
        self.in_layer = FCBlock(
            in_features=pos_enc_embedding_size, out_features=hidden_size
        )

        hidden1 = [
            FCBlock(in_features=hidden_size, out_features=hidden_size)
            for _ in range(hidden_layers_block)
        ]
        self.mid1 = torch.nn.Sequential(*hidden1)

        self.cat_layer = FCBlock(
            in_features=hidden_size + pos_enc_embedding_size, out_features=hidden_size
        )

        hidden2 = [
            FCBlock(in_features=hidden_size, out_features=hidden_size)
            for _ in range(hidden_layers_block)
        ]
        self.mid2 = torch.nn.Sequential(*hidden2)

        self.out_alpha = torch.nn.Linear(hidden_size, 1)

        self.apply(init_weights)

    def forward(
        self,
        x_pe: torch.tensor,
        noise_std: Optional[torch.tensor] = None,
        pe_mask: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        """
        x_pe is the position-encoded version of 3D points
        """
        # x_pe = self.positional_encoding(x)
        if pe_mask is not None:
            x_pe = torch.mul(x_pe, pe_mask)

        fc1 = self.in_layer(x_pe)
        fc2 = self.mid1(fc1)
        fc2_x = torch.cat((fc2, x_pe), dim=-1)
        fc3 = self.cat_layer(fc2_x)
        fc4 = self.mid2(fc3)
        raw = self.out_alpha(fc4)

        if noise_std is not None:
            noise = torch.randn(raw.shape, device=x.device) * noise_std
            raw = raw + noise
        alpha = raw * self.scale_output

        return alpha.squeeze(-1)


class DeepSdfDecoderMap(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(DeepSdfDecoderMap, self).__init__()

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x

    def decode_sdf(
        self, queries: torch.tensor, latent_vector: torch.tensor
    ) -> torch.tensor:
        num_samples = queries.shape[0]
        if latent_vector is None:
            inputs = queries
        else:
            latent_repeat = latent_vector.expand(num_samples, -1)
            inputs = torch.cat([latent_repeat, queries], 1)
        sdf_values = self(inputs)
        return sdf_values

    @torch.no_grad()
    def query_on_grid(self, latent_vec, N=256, max_batch=32**3):
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)  # spanning -1 to 1

        overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
        samples = torch.zeros(N**3, 4)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N
        # samples are now in [0, 255] in all three dimensions

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[2]
        # samples are now in [-1, 1] in all three dimensions

        num_samples = N**3
        head = 0
        while head < num_samples:
            sample_subset = samples[
                head : min(head + max_batch, num_samples), 0:3
            ].cuda()
            inference_out = self.decode_sdf(
                queries=sample_subset, latent_vector=latent_vec
            )
            samples[head : min(head + max_batch, num_samples), 3] = (
                inference_out.squeeze(1).detach().cpu()
            )
            head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)
        print(f"[sdf_values] min: {sdf_values.min()} | max: {sdf_values.max()}")
        return sdf_values, samples

    @torch.no_grad()
    def get_zero_level_set_pc(
        self, latent_vec: torch.tensor, _trunc_del: float = 0.001
    ) -> torch.tensor:
        # TODO: make this more efficient
        sdf_values, xyz_dist = self.query_on_grid(latent_vec=latent_vec, N=256)
        sdf_surf_mask = torch.logical_and(
            xyz_dist[:, 3] > -_trunc_del, xyz_dist[:, 3] < _trunc_del
        )
        pt_samples_zerolvl = xyz_dist[sdf_surf_mask][:, :3]
        return pt_samples_zerolvl

    def load_state_dict_from_ckpt(self, filepath: Path):
        def _remove_data_parallel(old_state_dict):
            new_state_dict = OrderedDict()
            for k, v in old_state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            return new_state_dict

        data = torch.load(str(filepath))
        self.load_state_dict(_remove_data_parallel(data["model_state_dict"]))
        print(f"[DeepSDF] loaded ckpt @ epoch {data['epoch']}: {filepath}")


class LatentCodes:
    def __init__(self, ckpt_path: Path, device: torch.device):
        self.device = device
        self.latent_codes = self._load_latent_codes_from_filepath(ckpt_path).to(device)

    def __len__(self):
        return self.latent_codes.size()[0]

    def __getitem__(self, idx):
        return self.latent_codes[idx]

    @staticmethod
    def _load_latent_codes_from_filepath(filepath: Path):
        data = torch.load(str(filepath))
        if isinstance(data, torch.Tensor):
            return data.cuda()
        elif isinstance(data["latent_codes"], torch.Tensor):
            num_vecs = data["latent_codes"].size()[0]
            lat_vecs = []
            for i in range(num_vecs):
                lat_vecs.append(data["latent_codes"][i].cuda())
            return lat_vecs
        else:
            num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape
            lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)
            lat_vecs.load_state_dict(data["latent_codes"])
            return lat_vecs.weight.data.detach()


if __name__ == "__main__":
    pe = PositionalEncoding(min_deg=0, max_deg=5, scale=1, transform=None)
    sdf = IsdfMap(
        pos_enc_embedding_size=pe.embedding_size,
        hidden_size=256,
        hidden_layers_block=1,
        scale_output=1,
    )

    x = torch.randn(4, 3)
    x = pe(x)
    o = sdf(x)
    print(o.shape)
    assert o.shape == (4,)
