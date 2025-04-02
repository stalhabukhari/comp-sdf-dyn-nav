import time
from pathlib import Path
from typing import Dict, Callable, Union

import numpy as np
import torch
from torch.autograd import grad
from rich import print


def start_timing():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.perf_counter()
        end = None
    return start, end


def end_timing(start, end):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()  # TODO: is this needed?
        elapsed_time = start.elapsed_time(end)
    else:
        end = time.perf_counter()
        elapsed_time = end - start
        # Convert to milliseconds to have the same units
        # as torch.cuda.Event.elapsed_time
        elapsed_time = elapsed_time * 1000
    return elapsed_time


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return points_grad


def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def load_checkpoint(
    model: Union[torch.nn.Module, torch.optim.Optimizer], ckpt_filepath: Path
):
    model.load_state_dict(state_dict=torch.load(ckpt_filepath))
    print(f" > Checkpoint Loaded: {ckpt_filepath}")


def save_checkpoint(
    model: Union[torch.nn.Module, torch.optim.Optimizer], ckpt_filepath: Path
):
    torch.save(model.state_dict(), ckpt_filepath)
    print(f" > Checkpoint Saved: {ckpt_filepath}")


def save_state_dicts(state_dicts: Dict[str, dict], filepath: Path):
    torch.save(state_dicts, filepath)
    print(f" > StateDicts Saved: {filepath}")


def chunks(
    pc: torch.tensor, chunk_size: int, fc_sdf_map: Callable, to_cpu: bool = False
) -> torch.tensor:
    n_pts = pc.shape[0]
    n_chunks = int(np.ceil(n_pts / chunk_size))
    alphas = []
    for n in range(n_chunks):
        start = n * chunk_size
        end = start + chunk_size
        chunk = pc[start:end, :]
        alpha = fc_sdf_map(chunk)
        alpha = alpha.squeeze(dim=-1)
        if to_cpu:
            alpha = alpha.cpu()
        alphas.append(alpha)
    alphas = torch.cat(alphas, dim=-1)
    return alphas


def chunked_map(
    map: Callable, pt3ds: torch.Tensor, chunk_size: int = 100000, **map_kwargs
) -> torch.Tensor:
    n_chunks = int(np.ceil(pt3ds.shape[0] / chunk_size))
    pred_full = torch.zeros(pt3ds.shape[0]).to(pt3ds.device)
    for n in range(n_chunks):
        pt3d_chunk = pt3ds[n * chunk_size : (n + 1) * chunk_size, :]
        pred_chunk = map(pt3d_chunk, **map_kwargs).squeeze(dim=-1)
        pred_full[n * chunk_size : (n + 1) * chunk_size] = pred_chunk
    return pred_full
