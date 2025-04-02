import os, random
from typing import Dict, Optional, Tuple, List, Union, Callable
from uuid import uuid4

import numpy as np
import pybullet as pb
from rich import print

from igibson.envs.igibson_env import iGibsonEnv
from igibson.external.pybullet_tools.utils import quat_from_euler
from igibson.objects.articulated_object import URDFObject
from igibson.robots import Turtlebot
from igibson.utils.assets_utils import get_ig_model_path, get_ig_avg_category_specs
from igibson.utils import transform_utils as ig_tformutil
from igibson.external.pybullet_tools.utils import (
    get_base_values,
    set_base_values,
    pairwise_collision,
)

from data import common as cmn
from data import image_utils as imutils


def initialize_igibson_env(cfg: Dict, debug: bool = False) -> iGibsonEnv:
    env = iGibsonEnv(
        config_file=cfg,
        mode="gui_interactive",
        action_timestep=1.0 / 120.0,
        physics_timestep=1.0 / 120.0,
        use_pb_gui=debug,
    )
    env.simulator.viewer.initial_pos = [1.0, 0.7, 1.7]
    env.simulator.viewer.initial_view_direction = [-0.5, -0.9, -0.5]
    env.simulator.viewer.reset_viewer()
    return env


def initialize_robot(
    env: iGibsonEnv, pos: Tuple = (0, 0, 0), orn: Tuple = (0, 0, 0)
) -> Turtlebot:
    assert (
        len(env.robots) == 1
    ), f"Expected one robot in scene, found: {len(env.robots)}"
    robot = env.robots[0]
    # land robot onto the floor, with given position and orientation
    env.land(robot, pos, orn)
    # robot.tuck()
    return robot


def get_object_by_body_id(env: iGibsonEnv, body_id: int):
    object_urdf = env.scene.objects_by_id[body_id]
    return object_urdf


def place_object_in_scene(
    env: iGibsonEnv, category: str, model: str, pos: Tuple, orn: Optional[Tuple]
) -> URDFObject:
    """
    orn is expected to be an euler (and not a quaternion)
    """
    avg_category_spec = get_ig_avg_category_specs()

    # Create the full path combining the path for all models and the name of the model
    model_path = get_ig_model_path(category, model)
    filename = os.path.join(model_path, model + ".urdf")

    # Create a unique name for the object instance
    obj_name = "{}_{}".format(category, uuid4())

    # Create and import the object
    sim_obj = URDFObject(
        filename,
        name=obj_name,
        category=category,
        model_path=model_path,
        avg_obj_dims=avg_category_spec.get(category),
        fit_avg_dim_volume=True,
        texture_randomization=False,
        overwrite_inertial=True,
    )
    # sim_obj.scale_object()
    env.simulator.import_object(sim_obj)
    # I have was having issues with placing multiple objects in the scene using the env.land method.
    # Instead, using obj.set_position_orientation works fine, with some additional iters of stepping sim.
    sim_obj.set_position_orientation(pos, quat_from_euler(orn))
    # env.land(sim_obj, pos, orn)
    return sim_obj


def get_object_pose_ig(ig_object):
    pos, orn = ig_object.get_position_orientation()
    return pos, orn


def get_object_pose(object_id: int) -> np.ndarray:
    """Not used"""
    pos, orn = pb.getBasePositionAndOrientation(object_id)
    c2w_transform = ig_tformutil.pose2mat(pose=(pos, orn))
    c2w_transform = c2w_transform
    return c2w_transform


def get_obstacles_in_env(env: iGibsonEnv) -> List:
    assert (
        len(env.robots) == 1
    ), f"Expected only one object in env, found: {len(env.robots[0])}"
    robot = env.robots[0]
    floor_id = env.scene.objects_by_category["floors"][0].get_body_ids()[0]
    obstacles = []
    for body_id in env.scene.get_body_ids():
        if body_id not in robot.get_body_ids() and body_id != floor_id:
            obstacles.append(body_id)
    return obstacles


def get_obstacles_but_walls_in_env(env: iGibsonEnv) -> List:
    assert (
        len(env.robots) == 1
    ), f"Expected only one object in env, found: {len(env.robots[0])}"
    robot = env.robots[0]
    floor_id = env.scene.objects_by_category["floors"][0].get_body_ids()[0]

    wall_ids = []
    for wall in env.scene.objects_by_category["walls"]:
        wall_ids += wall.get_body_ids()
    # add ceiling to wall as well
    for ceiling in env.scene.objects_by_category["ceilings"]:
        wall_ids += ceiling.get_body_ids()

    obstacles = []
    for body_id in env.scene.get_body_ids():
        if (
            body_id not in robot.get_body_ids()
            and body_id != floor_id
            and body_id not in wall_ids
        ):
            obstacles.append(body_id)
    return obstacles


def get_walls_in_env(env: iGibsonEnv) -> List:
    wall_ids = []
    for wall in env.scene.objects_by_category["walls"]:
        wall_ids += wall.get_body_ids()
    # add ceiling to wall as well
    for ceiling in env.scene.objects_by_category["ceilings"]:
        wall_ids += ceiling.get_body_ids()
    return wall_ids


def get_collision_fn_pb_no_shift(
    env: iGibsonEnv, obstacles: List, dist_thresh: float = 1e-6
) -> Callable:
    assert (
        len(env.robots) == 1
    ), f"Expected only one object in env, found: {len(env.robots[0])}"
    robot_body_ids = env.robots[0].get_body_ids()
    assert (
        len(robot_body_ids) == 1
    ), f"Only single body is supported, found: {len(robot_body_ids)}"
    robot_body_id = robot_body_ids[0]

    def _collision_fn_pb() -> bool:
        in_collision = any(
            pairwise_collision(robot_body_id, obs, max_distance=dist_thresh)
            for obs in obstacles
        )
        return in_collision

    return _collision_fn_pb


def get_collision_fn_pb_obs2obs(
    obs_to_check_id: int, other_obstacles: List, dist_thresh: float = 1e-6
) -> Callable:
    def _collision_fn_pb(_config: Union[List, Tuple]) -> bool:
        start_conf = get_base_values(obs_to_check_id)
        set_base_values(obs_to_check_id, _config)
        in_collision = any(
            pairwise_collision(obs_to_check_id, obs, max_distance=dist_thresh)
            for obs in other_obstacles
        )
        set_base_values(obs_to_check_id, start_conf)  # undo shift
        return in_collision

    return _collision_fn_pb


def random_obstacle_displacement(
    env: iGibsonEnv,
    obs_list: List,
    obs_static_list: List,
    obs_id_to_planar_radius: Dict,
    robot_state: List,
    forbidden_locs: List = [],
):
    scene_bounds = [
        [-1.7, 1.7],  # x
        [-3.6, 1],  # y
    ]  # TODO: add scene_bounds as func arg
    _euclidean_dist = lambda vec1, vec2: sum(
        [(vec1[_idx] - vec2[_idx]) ** 2 for _idx in range(3)]
    ) ** (1 / 2)

    # select obs to move
    obs_to_move_idx = random.randint(0, len(obs_list) - 1)
    obs_to_move_id = obs_list[obs_to_move_idx]
    obs_to_move_obj = env.scene.objects_by_id[obs_to_move_id]

    pos, orn = obs_to_move_obj.get_base_link_position_orientation()

    # get collision checker obs to obs (objects should be apart at least 0.8, to allow bot)
    obs2obs_clearance = 0.5  # 0.8
    other_obstacles = [_id for _id in obs_list if _id != obs_to_move_id]
    coll_checker_obs2obs = get_collision_fn_pb_obs2obs(
        obs_to_check_id=obs_to_move_id,
        other_obstacles=other_obstacles + obs_static_list,
        dist_thresh=obs2obs_clearance,
    )

    do_displace = True
    safety_dist = obs_id_to_planar_radius[obs_to_move_id]
    for _ in range(30000):
        # select displacement
        disp_xy = [
            random.uniform(0.2, 2) * (-1) ** random.randint(0, 1),
            random.uniform(0.2, 2) * (-1) ** random.randint(0, 1),
            0,
        ]

        new_pos = pos + np.array(disp_xy, dtype=np.float32)

        # check if it violates scene bounds
        if (
            new_pos[0] < scene_bounds[0][0]
            or new_pos[0] > scene_bounds[0][1]
            or new_pos[1] < scene_bounds[1][0]
            or new_pos[1] > scene_bounds[1][1]
        ):
            continue

        # check if new pos is in robot's memory regions
        if (
            _euclidean_dist(new_pos, robot_state) - safety_dist < 0.5
        ):  # static memory region
            # invalid
            continue

        # check if new pos is near forbidden locs
        if any(
            [
                _euclidean_dist(new_pos, _loc) - safety_dist < 0.3
                for _loc in forbidden_locs
            ]
        ):
            # invalid
            continue

        # check obstacle overlap
        is_collision = coll_checker_obs2obs(new_pos)
        if not is_collision:
            break
    else:
        print(
            "[Obs-Disp] Failure: Could not find a valid displacement for chosen obstacle"
        )
        do_displace = False

    if do_displace:
        # apply displacement
        pos[:2] += disp_xy[:2]
        obs_to_move_obj.set_base_link_position_orientation(pos, orn)
        print("[Obs-Disp] Success: Obstacle displaced!")
        cmn.OBS_DISP_COUNT += 1
