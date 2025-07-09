"""This script is used to test the static scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import math
import os
from typing import Literal

import rootutils
import torch
import tyro
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from get_started.utils import ObsSaver
from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.state import TensorState

def extract_observation(obs: TensorState) -> dict[str, torch.Tensor]:
    """Extract robot observation from the TensorState."""
    observation = {}
    from torchvision.utils import make_grid
    #extract image
    for camera_name, camera in obs.cameras.items():
        rgb_data = camera.rgb
        image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))
        observation[camera_name] = image
    #extract robot joint positions
    observation['states'] = next(iter(obs.robots.values())).body_state

    return observation
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
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

# add objects
scenario.objects = [
    PrimitiveCubeCfg(
        name="cube",
        size=(0.1, 0.1, 0.1),
        color=[1.0, 0.0, 0.0],
        physics=PhysicStateType.RIGIDBODY,
    ),
    PrimitiveCubeCfg(
        name="cube_2",
        size=(0.1, 0.1, 0.1),
        color=[0.0, 1.0, 0.0],
        physics=PhysicStateType.RIGIDBODY,
    )
]


# log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            "cube": {
                "pos": torch.tensor([0.0, 0.0, 0.05]),
                "rot": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            },
            "cube_1": {
                "pos": torch.tensor([0.0, 0.0, -0.05]),
                "rot": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            }
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
            "kinova_gen3_robotiq_2f85": {
                "pos": torch.tensor([0.0, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {
                    "joint_1": 0.0,
                    "joint_2": math.pi / 6,
                    "joint_3": 0.0,
                    "joint_4": math.pi / 2,
                    "joint_5": 0.0,
                    "joint_6": 0.0,
                    "joint_7": 0.0,
                    "finger_joint": 0.0,
                },
            },
        },
    }
    for _ in range(args.num_envs)
]


robot = scenario.robots[0]
# log.info(f"Robot: {robot.name}")
# log.info(f"Robot DOF: {robot.num_joints}")
#log.info(f"Robot joint names: {robot.joint_names}")
# log.info(f"Robot actuator names: {robot.actuators.keys()}")
# log.info(f"Robot parameters:{robot}")
*_, robot_ik = get_curobo_models(robot)

# log.info(f"Robot IK: {robot_ik.robot_config}")

curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)

# log.info(f"Robot number of DOF: {curobo_n_dof}")
ee_n_dof = len(robot.gripper_open_q)
# log.info(f"Robot end-effector DOF: {ee_n_dof}")

obs, extras = env.reset(states=init_states)
os.makedirs("get_started/output", exist_ok=True)


## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/test_pos_quart_{args.sim}.mp4")
obs_saver.add(obs)

step = 0
robot_joint_limits = scenario.robots[0].joint_limits
ee_pos = obs.robots[robot.name].body_state[0][0][:3]
ee_quat = obs.robots[robot.name].body_state[0][0][3:7]
log.info(f"Initial end-effector position: {ee_pos}")
log.info(f"Initial end-effector quaternion: {ee_quat}")
for step in range(200):

    # dummy_actions = [
    #     {robot.name: {"dof_pos_target": dict(zip(robot.actuators.keys(), [0.0] * robot.num_joints))}}
    #     for _ in range(scenario.num_envs)
    # ]
    # if step == 0:
    #     for _ in range(50):
    #         obs, _, _, _, _ = env.step(dummy_actions)
    #         obs_saver.add(obs)

    obs_saver.add(obs)
    states = env.handler.get_states()

    curr_robot_q = states.robots[robot.name].joint_pos.cuda()
    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

    # len_action_chunk = 10
    # action_dim = 7
    # action_chunk = [torch.zeros((len_action_chunk,action_dim), device="cuda:0") for _ in range(scenario.num_envs)]

    ee_pos_target = ee_pos.unsqueeze(0).repeat(scenario.num_envs, 1).to("cuda:0") + torch.Tensor([0 + 0.1*(step/100),0,0]).unsqueeze(0).repeat(scenario.num_envs, 1).to("cuda:0")
    log.debug(f"Step {step}, End-effector target position: {ee_pos_target}")
    ee_quat_target = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0]] * scenario.num_envs,
        device="cuda:0",
    )
    result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

    q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda:0")
    ik_succ = result.success.squeeze(1)
    q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    q[:, -ee_n_dof:] = 0.04
    robot = scenario.robots[0]

    actions = [
        {robot.name: {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))}}
        for i_env in range(scenario.num_envs)
    ]

    # if step % 50 == 0:
    #     log.debug(f"Step {step}, Actions: {actions}")

    obs, reward, success, time_out, extras = env.step(actions)


    step += 1

obs_saver.save()
