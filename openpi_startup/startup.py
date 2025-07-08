from __future__ import annotations
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import rootutils
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from scipy.spatial.transform import Rotation as R

from get_started.utils import ObsSaver
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils import configclass
from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.setup_util import get_sim_env_class


@configclass
class Args:
    """Arguments for the static scene."""
    robot: str = "franka"
    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco"] = "isaaclab"
    ## Others
    num_envs: int = 1
    headless: bool = False
    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")

args = tyro.cli(Args)

# initialize scenario
scenario = ScenarioCfg(
    robots=[args.robot],
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

"""scenario设置"""

"""返回环境接口"""
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)


"""设置初始状态init states"""
init_states =  []

"""机器人型号读取、IK模型获取、关节数获取、末端执行器关节数获取"""
robot = scenario.robots[0]
*_, robot_ik = get_curobo_models(robot)
curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
ee_n_dof = len(robot.gripper_open_q)

"""observation保存"""
obs, extras = env.reset(states=init_states)
os.makedirs("get_started/output", exist_ok=True)

## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/motion_planning/0_franka_planning_{args.sim}.mp4")
obs_saver.add(obs)

"""移动抓取"""
def move_to_pose(
    obs, obs_saver, robot_ik, robot, scenario, ee_pos_target, ee_quat_target, steps=10, open_gripper=False
):
    """Move the robot to the target pose."""
    curr_robot_q = obs.robots[robot.name].joint_pos

    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

    result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

    q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda:0")
    ik_succ = result.success.squeeze(1)
    q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    q[:, -ee_n_dof:] = 0.04 if open_gripper else 0.0
    actions = [
        {robot.name: {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))}}
        for i_env in range(scenario.num_envs)
    ]
    for i in range(steps):
        obs, reward, success, time_out, extras = env.step(actions)
        obs_saver.add(obs)
    return obs

"""主循环"""
def run_loop(steps = 0):
    robot_joint_limits = scenario.robots[0].joint_limits
    for step in range(steps):
        log.debug(f"step {step}")
        states = env.handler.get_states()
        curr_robot_q = states.robots[robot.name].joint_pos.cuda()
        seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])
        """设置末端执行器目标位置和姿态"""
        ee_pos_target = torch.zeros((args.num_envs, 3), device="cuda:0")
        ee_quat_target = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]] * args.num_envs,
            device="cuda:0",
        )
        obs = move_to_pose(
            obs, obs_saver, robot_ik, robot, scenario, ee_pos_target, ee_quat_target, steps=steps, open_gripper=True
        )
        step += 1

if __name__ == "__main__":
    log.info("Starting the static scene test.")
    fake_obs = env.reset(states=init_states)
    log.info(f"Fake observation: {fake_obs}")
    fake_actions = []
