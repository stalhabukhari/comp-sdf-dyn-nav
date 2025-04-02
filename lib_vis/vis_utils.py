from typing import List, Callable

import numpy as np
import skimage
import torch
import trimesh
import matplotlib.pyplot as plt


def marching_cubes_trimesh(
    numpy_3d_sdf_tensor: np.ndarray, level=0.0
) -> trimesh.Trimesh:
    """Convert sdf samples to triangular mesh."""
    vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level
    )
    dim = numpy_3d_sdf_tensor.shape[0]
    vertices = vertices / (dim - 1)
    mesh = trimesh.Trimesh(
        vertices=vertices, vertex_normals=vertex_normals, faces=faces
    )
    return mesh


def draw_mesh(
    sdf: torch.tensor,
    scale: float = None,
    transform: np.ndarray = None,
    color_by: str = "normals",
) -> trimesh.Trimesh:
    """Run marching cubes on sdf tensor to return mesh."""
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()
    mesh = marching_cubes_trimesh(sdf)

    # Transform to [-1, 1] range
    mesh.apply_translation([-0.5, -0.5, -0.5])
    mesh.apply_scale(2)

    # Transform to scene coordinates
    if scale is not None:
        mesh.apply_scale(scale)
    if transform is not None:
        mesh.apply_transform(transform)

    if color_by == "normals":
        norm_cols = (-mesh.vertex_normals + 1) / 2
        norm_cols = np.clip(norm_cols, 0.0, 1.0)
        norm_cols = (norm_cols * 255).astype(np.uint8)
        alphas = np.full([norm_cols.shape[0], 1], 255, dtype=np.uint8)
        cols = np.concatenate((norm_cols, alphas), axis=1)
        mesh.visual.vertex_colors = cols
    elif color_by == "height":
        zs = mesh.vertices[:, 1]
        cols = trimesh.visual.interpolate(zs, color_map="viridis")
        mesh.visual.vertex_colors = cols
    else:
        mesh.visual.face_colors = [160, 160, 160, 255]

    return mesh


def sdf_contour_plot(
    sdf_fn: Callable,
    x_lims: List[float],
    y_lims: List[float],
    grid_res: int = 100,
    contour_levels: int = 20,
):
    x_axis = np.linspace(*x_lims, num=grid_res)
    y_axis = np.linspace(*y_lims, num=grid_res)

    contours = np.zeros((y_axis.size, x_axis.size), dtype=np.float32)
    for x_idx in range(x_axis.size):
        for y_idx in range(y_axis.size):
            contours[y_idx, x_idx] = sdf_fn(x_axis[x_idx], y_axis[y_idx])
    contours = np.array(contours)
    # plots contour lines
    fig, ax = plt.subplots(1, 1)
    ax.contourf(x_axis, y_axis, contours, levels=contour_levels)
    ax.set_title("Contour Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


def point_cloud_show(pcd: np.ndarray):
    pcd = trimesh.PointCloud(pcd)
    trimesh.Scene(pcd).show()
