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


from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class
from scene.utils import ObsSaver

import collections
import dataclasses
import logging
import pathlib
import imageio
import numpy as np
import tqdm
import tyro
import torch

# 2. 导入 openpi 客户端
from openpi_client import websocket_client_policy as _websocket_client_policy

# from pick_up_middle_bottom import scenario, init_states
from curobo.types.math import Pose
from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.setup_util import get_sim_env_class
from pi_utils import extract_observation

ROBOVERSE_DUMMY_ACTION = [0.0] * 7 + [0.0]
ROOAVERSE_ENV_RESOLUTION = 512

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int = 10
    env_id: str = "Roboverse"  # 您的环境ID
    max_steps: int = 200  # 单个回合的最大步数
    num_trials_per_task: int = 3  # 每个任务的尝试次数
    num_steps_wait: int = 15
    video_out_path: str = "data/roboverse/videos"
    seed: int = 42
"""This script is used to test the static scene."""



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

# add cameras
scenario.cameras = [
    PinholeCameraCfg("camera0", width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0)),
    PinholeCameraCfg("camera1", width=1024, height=1024, pos=(0.3, 0.0, 2.5), look_at=(0.3, 0.0, 0.0)),
]

# add objects
scenario.objects = [
    RigidObjCfg(
        name="bbq_sauce_1_1",
        scale=(2, 2, 2),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="scene/example_assets/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="scene/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
        mjcf_path="scene/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
    ),
    RigidObjCfg(
        name="bbq_sauce_1_2",
        scale=(2, 2, 2),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="scene/example_assets/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="scene/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
        mjcf_path="scene/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
    ),
    RigidObjCfg(
        name="bbq_sauce_1_3",
        scale=(2, 2, 2),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="scene/example_assets/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="scene/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
        mjcf_path="scene/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
    ),
]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            "bbq_sauce_1_1": {
                "pos": torch.tensor([0.4, -0.3, 0.13]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "bbq_sauce_1_2": {
                "pos": torch.tensor([0.4, 0.0, 0.13]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "bbq_sauce_1_3": {
                "pos": torch.tensor([0.4, 0.3, 0.13]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
        },
        "robots": {
            "franka": {
                "pos": torch.tensor([0.0, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.570796,
                    "panda_joint7": 0.785398,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
            },
        },
    }
]

def eval_roboverse(args: Args) -> None:

    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    env_class = get_sim_env_class(scenario.sim)
    env = env_class(scenario)
    task_description = "pick up the middle bbq sauce bottle"

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    robot = get_curobo_models(scenario.robots)[0]
    if not robot:
        raise ValueError("No robots found in the scenario. Please check your scenario configuration.")
    *_, robot_ik = get_curobo_models(robot)
    curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
    ee_n_dof = len(robot.gripper_open_q)
    robot_joint_limits = scenario.robots[0].joint_limits

    total_episodes, total_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task: {task_description}"):
        logging.info(f"\nTask: {task_description}")
        obs, extras = env.reset(states=init_states)
        action_plan = collections.deque()
        # 设置
        t = 0
        replay_images = []
        logging.info(f"Starting episode {episode_idx + 1}...")
        while t < args.max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(ROBOVERSE_DUMMY_ACTION)
                    t += 1
                    continue

                replay_images.append(obs["camera0"])
                states = env.handler.get_states()
                curr_robot_q = states.robots[robot.name].joint_pos.cuda()
                seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

                if not action_plan:

                    element = extract_observation(obs)
                    element["task_description"] = task_description
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(action_chunk[: args.replan_steps])
                action = action_plan.popleft()
                logging.info(f"Step {t}, Action: {action}")
                result = robot_ik.solve_batch(Pose(action[:,7].repeat(scenario.num_envs, 1)), seed_config=seed_config)
                q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda:0")
                ik_succ = result.success.squeeze(1)
                q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
                q[:, -ee_n_dof:] = 0.04
                actions = [
                            {robot.name: {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))}}
                            for i_env in range(scenario.num_envs)
                        ]

                obs, reward, success, time_out, extras = env.step(actions) # 无需 .tolist() 如果 action 已经是 list/numpy

            except Exception as e:
                logging.error(f"Caught exception: {e}", exc_info=True)
                break



        suffix = "test"
        task_segment = args.env_id.replace("/", "_")
        video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4"
        imageio.mimwrite(video_path, replay_images, fps=15, quality=8)
        logging.info(f"Saved video to {video_path}")

        # 打印日志
        logging.info(f"Success rate so far: {total_successes / total_episodes * 100:.1f}% ({total_successes}/{total_episodes})")

    logging.info(f"Final success rate: {total_successes / total_episodes * 100:.1f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_roboverse)
