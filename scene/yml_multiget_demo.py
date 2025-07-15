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
config_path = f"scene/config/yml_multigen_demo.yml"
scenario, init_states = get_scene_from_config(config_path)

# setup the initial state
log.info(f"Using simulation type: {scenario.sim}")
env_class = get_sim_env_class(SimType(scenario.sim))
scene_env = env_class(scenario=scenario)
log.info(f"Initial states for each environment: {init_states}")
obs, extras = scene_env.reset(states=init_states)

os.makedirs("scene/output", exist_ok=True)
output_save_path = f"scene/output/yml_multiget_demo_{args.sim}.png"
log.info(f"Saving image to {output_save_path}")
imageio.imwrite(output_save_path, next(iter(obs.cameras.values())).rgb[0].cpu().numpy())
