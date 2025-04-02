import numpy as np
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots import Turtlebot
from igibson.utils import transform_utils as ig_tformutil

from data import dataset as ds
from data import image_utils as imutils

_EPS = np.finfo(float).eps * 4.0


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return homogeneous rotation matrix from quaternion.
    ref: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/transformation.py#L1043
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    mat_h = np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return mat_h


def get_current_robot_camera_pose(env: iGibsonEnv) -> np.ndarray:
    assert len(env.robots) == 1, f"Found {len(env.robots)} robot instances in env"
    robot: Turtlebot = env.robots[0]
    camera_pos = robot.eyes.get_position()
    camera_orn = robot.eyes.get_orientation()
    c2w_transform = ig_tformutil.pose2mat(pose=(camera_pos, camera_orn))

    rot_y = np.array(
        [
            [np.cos(np.pi / 2), 0, np.sin(np.pi / 2), 0],
            [0, 1, 0, 0],
            [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2), 0],
            [0, 0, 0, 1],
        ]
    ).astype(np.float32)
    rot_z = np.array(
        [
            [np.cos(-np.pi / 2), -np.sin(-np.pi / 2), 0, 0],
            [np.sin(-np.pi / 2), np.cos(-np.pi / 2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).astype(np.float32)
    c2w_transform = c2w_transform @ rot_y @ rot_z
    return c2w_transform


def get_current_robot_rgbd_framedata(env: iGibsonEnv) -> ds.RGBDFrameData:
    state = env.get_state()
    # image
    rgb_frame = state["rgb"]
    assert (
        len(rgb_frame.shape) == 3 and rgb_frame.shape[-1] == 3
    ), f"Unexpected RGB frame shape: {rgb_frame.shape}"
    rgb_img = imutils.cvt_img_float_to_uint8(img=rgb_frame)
    # depth
    depth_img = np.squeeze(state["depth"])
    depth_img *= env.sensors["vision"].depth_high
    # c2w transform (robot camera)
    c2w_transform_mat = get_current_robot_camera_pose(env=env)
    # create data structure
    metadata = ds.RGBDFrameMetadata(
        rgb_filepath="", depth_filepath="", transform_c2w=c2w_transform_mat.tolist()
    )
    frame_data = ds.RGBDFrameData(
        rgb_img=rgb_img, depth_img=depth_img, metadata=metadata
    )
    return frame_data
