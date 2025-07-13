"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
import imageio
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
from scene.scene_config_utils import gen_scene_to_config, get_scene_from_config


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
scenario.cameras = [
    PinholeCameraCfg("cam1", width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0)),
    PinholeCameraCfg("cam2", width=1024, height=1024, pos=(0.3, 0.0, 2.5), look_at=(0.3, 0.0, 0.0)),
]

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
                        z,
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
os.makedirs("scene/config", exist_ok=True)
output_save_path = f"scene/output/yml_gen_demo_{args.sim}.png"
config_save_path = f"scene/config/yml_gen_demo.yml"
log.info(f"Saving image to {output_save_path}")
imageio.imwrite(output_save_path, next(iter(obs.cameras.values())).rgb[0].cpu().numpy())
gen_scene_to_config(scenario, init_states, config_save_path)
log.info(f"Configuration saved to {config_save_path}")
