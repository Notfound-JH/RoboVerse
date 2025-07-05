"""This script is used to test the static scene."""

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


from scene.utils import ObsSaver
from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco"] = "isaaclab"

    ## Others
    num_objs: int = 2
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
scenario.cameras = [PinholeCameraCfg("cam1", width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0)),
                    PinholeCameraCfg("cam2", width=1024, height=1024, pos=(0.3, 0.0, 2.5), look_at=(0.3, 0.0, 0.0))]

# add objects
scenario.objects = [
    RigidObjCfg(
        name=f"bbq_sauce_{i}",
        scale=(2, 2, 2),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="scene/example_assets/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="scene/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
        mjcf_path="scene/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
    )
    for i in range(1, args.num_objs + 1)
]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

# zone
x_low, x_high = 0.3, 1.0
y_low, y_high = -0.5, 0.5
z = 0.13

init_states = [
    {
        "objects": {
            f"bbq_sauce_{i}": {
                "pos": torch.tensor(
                    [
                        x_low + (x_high - x_low) * torch.rand(1).item(),
                        y_low + (y_high - y_low) * torch.rand(1).item(),
                        z
                    ],
                ),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            }
            for i in range(1, args.num_objs + 1)
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
obs, extras = env.reset(states=init_states)
os.makedirs("scene/output", exist_ok=True)


## Main loop
obs_saver = ObsSaver(video_path=f"scene/output/dynamic_random_scene_01_{args.sim}.mp4")
obs_saver.add(obs)

step = 0
robot = scenario.robots[0]
for _ in range(100):
    log.debug(f"Step {step}")
    actions = [
        {
            robot.name: {
                "dof_pos_target": {
                    joint_name: (
                        torch.rand(1).item() * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                        + robot.joint_limits[joint_name][0]
                    )
                    for joint_name in robot.joint_limits.keys()
                }
            }
        }
        for _ in range(scenario.num_envs)
    ]
    obs, reward, success, time_out, extras = env.step(actions)
    obs_saver.add(obs)
    step += 1

obs_saver.save()
