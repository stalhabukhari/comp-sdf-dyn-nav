"""
python sim_robot_sdf.py <path-to-config>
"""

import shutil
import time
from pathlib import Path
from typing import Dict, Callable

import numpy as np
import torch
from rich import print

import pybullet as pb
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots import Turtlebot

from data import common as cmn
from data import structures as struct
from data import image_utils as imutils
from lib_igibson.igibson_env_utils import (
    initialize_igibson_env,
    initialize_robot,
    place_object_in_scene,
    get_obstacles_in_env,
    get_collision_fn_pb_no_shift,
    get_obstacles_but_walls_in_env,
    get_walls_in_env,
    random_obstacle_displacement,
)
from lib_igibson.igibson_img_utils import get_current_robot_rgbd_framedata
from lib_control.pure_pursuit_controller import (
    State,
    TargetCourse,
    proportional_control,
    pure_pursuit_steer_control,
)
from lib_plan import traj_utils
from infer_single_view import InferenceSingleViewAugSDF
from planner import TrajectoryOptimizerBodySDF


class ObjCompDynamicRobotSDFPlannerWithControl:
    def __init__(self, sdf_config: Dict, sim_config: Dict, robs_config: Dict):
        self.robot_state_cfg = robs_config["robot"]
        traj_opt_cfg = sdf_config["traj_opt"]
        self.sdf_inf = InferenceSingleViewAugSDF(config_dict=sdf_config)

        # for dynamic planning
        body_sdf_latent_code_filepath = (
            Path(sdf_config["model"]["ckpt_dir"])
            / sdf_config["model"]["sdf_object"]["ckpt_latent_robot_body"]
        )
        self.traj_opt = TrajectoryOptimizerBodySDF(
            sdf_infer_map=self.sdf_inf,
            body_sdf_latent_code_filepath=body_sdf_latent_code_filepath,
            iterations_per_update=50,  # traj_opt_cfg["opt_iters"],
            lr_init=traj_opt_cfg["lr_init"],
            lr_decay=traj_opt_cfg["lr_decay"],
            trunc_dist_sdf=traj_opt_cfg["safety_margin"],
            smoothing=traj_opt_cfg["smoothing_robot_sdf"],
            pc_memory_radius=sdf_config["memory"]["radius_outer"],
            sdf_scene_map=self.sdf_inf.sdf_stc_pe,
        )
        self.robot_grid_z_offset = traj_opt_cfg["robot_grid_center_from_ground"]
        self.robot_start = np.array(
            self.robot_state_cfg["state_initial"]["position"], dtype=np.float32
        )
        self.robot_goal = np.array(
            self.robot_state_cfg["state_final"]["position"], dtype=np.float32
        )
        self.torch_device = self.sdf_inf.device

        self._define_env(sim_config=sim_config, robs_config=robs_config)

    def _define_env(self, sim_config: Dict, robs_config: Dict):
        sim_config["load_object_categories"] = []
        sim_config["load_room_types"] = ["living_room"]
        sim_config["hide_robot"] = False
        sim_config["texture_randomization_freq"] = None
        sim_config["object_randomization_freq"] = None

        # setup env
        self.env = initialize_igibson_env(cfg=sim_config, debug=True)

        # add obstacles
        self.obstacles_ig = []
        for idx, obs_i in robs_config["obstacles"].items():
            obs_name = obs_i["name"]
            obs_inst = obs_i["instance"]
            obs_pose = obs_i["state"]
            _obstacle = place_object_in_scene(
                env=self.env,
                category=obs_name,
                model=obs_inst,
                pos=obs_pose["position"],
                orn=obs_pose["orientation"],
            )
            self.obstacles_ig.append(_obstacle)

        # setup robot
        robot_state = robs_config["robot"]["state_initial"]
        initialize_robot(
            env=self.env, pos=robot_state["position"], orn=robot_state["orientation"]
        )
        assert len(self.env.scene.robots) == 1
        self.robot = self.env.scene.robots[0]

        for _ in range(30):
            self.env.simulator.step()
        self.env.simulator.sync(force_sync=True)

        # get all obstacles:
        self.obstacles = get_obstacles_in_env(env=self.env)
        # collision function (pybullet-based)
        self.collision_fn = get_collision_fn_pb_no_shift(
            env=self.env, obstacles=self.obstacles, dist_thresh=1e-6
        )

        # set camera position
        pb.resetDebugVisualizerCamera(
            cameraDistance=2.5,
            cameraYaw=-40,
            cameraPitch=-40,
            cameraTargetPosition=[-0.2, -1.5, 0.5],
        )

    @staticmethod
    def get_trajectory_as_targetcourse(traj) -> TargetCourse:
        target_course = TargetCourse(traj[:, 0], traj[:, 1])
        return target_course

    def get_optimized_trajectory(
        self,
        frame_sample: struct.RGBDFrameSample,
        traj_init: np.ndarray,
        robot_state: State,
    ) -> np.ndarray:
        ## Preparation ##
        with torch.no_grad():
            traj_init = torch.from_numpy(traj_init).float()
        fs_buffer = self.sdf_inf.get_fs_buffer_from_frame_sample(frame_sample)
        # pc sampling
        sampling_mask = (
            (frame_sample.depth_img < self.traj_opt.pc_memory.radius)
            * (frame_sample.depth_img > 0.02)
        ).astype(bool)
        sampling_mask = torch.from_numpy(sampling_mask).to(self.torch_device)
        sampled_pc_pts = fs_buffer.get_surface_points_from_depth_batch(
            sampling_mask=sampling_mask
        )
        # obj_det_preds = self.sdf_inf.get_yolo_predictions(fs_buffer=fs_buffer)
        obj_det_preds = None

        # [robot-sdf] update pc memory
        robot_curr = np.array([robot_state.x, robot_state.y], dtype=np.float32)
        self.traj_opt.pc_memory.pre_update_setup(
            robot_state_new=robot_curr,
            cam_params=self.sdf_inf.cam_params,
            c2w_tform=frame_sample.transform,
        )

        ## TrajOpt ##
        traj_opted = self.traj_opt.optimize_trajectory(
            pc=sampled_pc_pts.pc.reshape(-1, 3),
            obj_det_preds=obj_det_preds,
            traj_pt2ds=traj_init,
            z_value=self.robot_grid_z_offset,
            check_traj_coll_free=False,
        )

        return traj_opted.numpy()

    def get_robot_rgbd_frame_sample(self) -> struct.RGBDFrameSample:
        frame_data = get_current_robot_rgbd_framedata(env=self.env)
        frame_sample = struct.RGBDFrameSample.from_rgbd_frame_data(
            frame_data=frame_data, device=self.torch_device
        )
        return frame_sample

    def simulation_step(
        self, target_course: TargetCourse, max_time_sec: float = 1.0
    ) -> State:
        state = get_robot_state(robot=self.robot)

        target_speed = 3  # << HYP
        target_ind, _ = target_course.search_target_index(state)

        time = 0
        while time < max_time_sec:
            # Calc control input
            pos_control = proportional_control(target_speed, state.v)
            theta_control, target_ind = pure_pursuit_steer_control(
                state, target_course, target_ind
            )
            control_action = (pos_control, theta_control)
            self.robot.apply_action(action=control_action)

            self.env.simulator.step()
            self.env.simulator.sync(force_sync=True)
            time += self.env.simulator.physics_timestep
            state = get_robot_state(robot=self.robot)
            # had to add stopping condition here as well
            if (
                state.calc_distance(
                    point_x=self.robot_goal[0], point_y=self.robot_goal[1]
                )
                < cmn.STOP_COND_DIST
            ):
                break
        return state

    def _definitions_for_dynamic_obs_displacements(self) -> Callable:
        assert cmn.DYNAMIC_OBS
        # ----------------------- setup for dynamic obstacle placement ----------------------
        coll_obs_dyn = get_obstacles_but_walls_in_env(env=self.env)
        wall_ids = get_walls_in_env(env=self.env)
        # pre-compute object extents (planar) to help object placements during dynamic sim
        object_id_to_extent_radius_map = dict()
        for obs_id in coll_obs_dyn:
            data_dict = self.sdf_inf.misc_data.get_object_category_instance_for_body_id(
                env=self.env, body_id=obs_id
            )

            object_category = data_dict["category"]
            object_instance = data_dict["instance"]
            print(
                f"[DEBUG] fetching data for object: {object_category} | {object_instance}"
            )

            obj_bbox_extents_scaled = (
                self.sdf_inf.misc_data.get_object_bbox_size_scaled(
                    object_category=object_category, object_model=object_instance
                )
            )
            max_obj_extent_xy = float(np.sqrt(np.sum(obj_bbox_extents_scaled[:2] ** 2)))
            object_id_to_extent_radius_map[obs_id] = max_obj_extent_xy

        def _rob_disp_fn(robot_state: State):
            random_obstacle_displacement(
                env=self.env,
                obs_list=coll_obs_dyn,
                obs_static_list=wall_ids,
                obs_id_to_planar_radius=object_id_to_extent_radius_map,
                robot_state=[robot_state.x, robot_state.y, 0.25],
                forbidden_locs=[self.robot_goal.tolist()],
            )
            self.env.simulator.sync(force_sync=True)
            self.env.simulator.step()
            self.env.simulator.sync(force_sync=True)

        return _rob_disp_fn

    @cmn.rstatus()
    def plan_control_loop(self, obs_move_step_size=200, sim_step_size=20):
        start_time = time.perf_counter()
        phys_timestep = self.env.simulator.physics_timestep
        target_course = None

        if cmn.DYNAMIC_OBS:
            # ----------------------- setup for dynamic obstacle placement ----------------------
            _rob_disp_fn = self._definitions_for_dynamic_obs_displacements()

        # ------------------------------------- Initialization ---------------------------------------
        traj = traj_utils.get_trajectory_straight_line(
            state_init=self.robot_start, state_fin=self.robot_goal
        )
        state = get_robot_state(robot=self.robot)
        # --------------------------------------------------------------------------------------------
        is_traj_too_short = False
        while True:
            if cmn.DYNAMIC_OBS:
                if (
                    cmn.COUNTER in [60, 160, 280]
                    or cmn.COUNTER % obs_move_step_size == 0
                ) and cmn.COUNTER > 0:
                    # random displacements of an obstacle
                    _rob_disp_fn(robot_state=state)

            if cmn.COUNTER % sim_step_size == 0:
                # ----------------------------------------- Planning ------------------------------------

                # check if trajectory is too short to do planning.
                if traj.shape[0] <= 3 and not is_traj_too_short:
                    # try re-init
                    traj = traj_utils.get_trajectory_straight_line(
                        state_init=np.array([state.x, state.y], dtype=np.float32),
                        state_fin=self.robot_goal,
                    )
                    target_course = self.get_trajectory_as_targetcourse(traj)
                    if traj.shape[0] <= 3:
                        # traj too short to optimize, keep doing control only (do not plan)
                        is_traj_too_short = True

                # continue planning if trajectory isn't too short
                if not is_traj_too_short:

                    if target_course is not None:
                        # discarding old traj points before planning
                        _traj_idx = min(
                            target_course.old_nearest_point_index, traj.shape[0] - 2
                        )
                        traj = traj[_traj_idx:]

                    # fetch current state
                    frame_sample = self.get_robot_rgbd_frame_sample()

                    # [dual-mode] traj-opt
                    traj = self.get_optimized_trajectory(
                        frame_sample=frame_sample, traj_init=traj, robot_state=state
                    )

                    target_course = self.get_trajectory_as_targetcourse(traj)
                    # debug
                    pb_draw_line(
                        traj, env=self.env, life_sec=phys_timestep * sim_step_size + 1.5
                    )

            # control
            state = self.simulation_step(
                target_course=target_course, max_time_sec=phys_timestep
            )
            cmn.COLLISIONS += int(self.collision_fn())

            # --------------------------------- Stopping conditions ---------------------------------
            if (
                state.calc_distance(
                    point_x=self.robot_goal[0], point_y=self.robot_goal[1]
                )
                < cmn.STOP_COND_DIST
            ):
                # success
                print("[Exiting]: Robot near goal (Success)")
                print(f"total obstacle displacements: {cmn.OBS_DISP_COUNT}")
                break
            if time.perf_counter() - start_time > cmn.STOP_COND_TIME:
                # failure
                print("[Exiting]: Plan/Control could not complete (Failure)")
                print(f"total obstacle displacements: {cmn.OBS_DISP_COUNT}")
                exit(0)
            if cmn.COLLISIONS > cmn.STOP_COND_COLL:
                # failure
                print("[Exiting]: Plan/Control could not complete (Failure)")
                print(f"total obstacle displacements: {cmn.OBS_DISP_COUNT}")
                exit(0)
            # ---------------------------------------------------------------------------------------

            cmn.COUNTER += 1


def pb_draw_line(
    traj2d: np.ndarray, env: iGibsonEnv, z_offset: float = 0.3, life_sec: float = 0
):
    traj3d = np.concatenate(
        (traj2d, np.zeros((traj2d.shape[0], 1)) + z_offset), axis=-1
    )
    for idx in range(traj3d.shape[0] - 1):
        this = traj3d[idx]
        next = traj3d[idx + 1]
        pb.addUserDebugLine(
            this.tolist(),
            next.tolist(),
            lineColorRGB=[0, 1, 0],
            lineWidth=2.0,
            lifeTime=life_sec,
            physicsClientId=env.simulator.cid,
        )


def get_robot_state(robot: Turtlebot):
    _x, _y, _z = robot.get_position()
    orn_quat = robot.get_orientation()
    vel_xyz = robot.get_linear_velocity()
    rpy = list(pb.getEulerFromQuaternion(orn_quat))
    state = State(x=_x, y=_y, yaw=rpy[2] + np.pi, v=np.linalg.norm(vel_xyz))
    state.rear_x = state.x
    state.rear_y = state.y
    return state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=Path, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Dynamic obstacle placement"
    )
    args = parser.parse_args()

    if args.dynamic:
        print("[INFO]: Dynamic obstacles enabled")
        cmn.DYNAMIC_OBS = args.dynamic

    robs_config = cmn.parse_yaml_file(filepath=args.cfg)
    sdf_config = cmn.parse_yaml_file(
        filepath=Path("configs/inference_single_view.yaml")
    )
    sim_config = cmn.parse_yaml_file(filepath=Path("configs/config-sim-turtlebot.yaml"))

    sim = ObjCompDynamicRobotSDFPlannerWithControl(
        sim_config=sim_config, sdf_config=sdf_config, robs_config=robs_config
    )
    sim.plan_control_loop()
