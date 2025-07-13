from __future__ import annotations

import os
import yaml
import rootutils
import torch
from typing import List
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils.setup_util import get_sim_env_class


def to_torch_tensor(data):
    """Convert data to a torch tensor."""
    if isinstance(data, (list, tuple)):
        return torch.tensor(data)
    elif isinstance(data, dict):
        return {k: to_torch_tensor(v) for k, v in data.items()}
    else:
        return torch.tensor(data)

def read_config(config_name: str) -> dict:
    with open(os.path.join(rootutils.find_root(), "scene","config", config_name), "r") as f:
        config = yaml.safe_load(f)
    import pprint
    pprint.pprint(config)
    return config

def get_scene_from_config(config_name: str) -> ScenarioCfg:
    """Read the scene configuration from a YAML file."""
    config = read_config(config_name)

    scenario_data = config['scenario']
    scenario = ScenarioCfg(
        robots=scenario_data.get('robots', [scenario_data['robots']]),
        try_add_table=scenario_data.get('try_add_table', False),
        sim=scenario_data.get('sim', 'isaaclab'),
        headless=scenario_data.get('headless', False),
        num_envs=scenario_data.get('num_envs', 1),
    )
    log.info(f"Scenario configuration loaded from {config_name}: {scenario}")
    # Add cameras
    scenario.cameras = [PinholeCameraCfg(**cam_data) for cam_data in scenario_data.get('cameras', [])]

    #Add objects
    for obj_data in scenario_data.get('objects', []):
        obj_data['physics'] = PhysicStateType[obj_data['physics']]
    scenario.objects = [RigidObjCfg(**obj_data) for obj_data in scenario_data.get('objects', [])]

    #setup the initial state
    if 'initial_state' in config:
        initial_state = config['initial_state']
        if 'robots' in initial_state:
            initial_state['robots'] =  {
                obj_name: to_torch_tensor(state) for obj_name, state in initial_state['robots'].items()
            }
        if 'objects' in initial_state:
            initial_state['objects'] = {
                obj_name: to_torch_tensor(state) for obj_name, state in initial_state['objects'].items()
            }
        if 'cameras' in initial_state:
            initial_state['cameras'] = {
                cam_name: to_torch_tensor(state) for cam_name, state in initial_state['cameras'].items()
            }
    init_states = [initial_state for _ in range(scenario.num_envs)]
    log.debug(f"Initial states for each environment: {init_states}")
    log.info(f"Using simulation type: {scenario.sim}")

    env_class = get_sim_env_class(SimType(scenario.sim))
    scene_env = env_class(scenario=scenario)
    init_obs,init_extras = scene_env.reset(states = init_states)

    return scene_env, init_obs, init_extras

def gen_scene_to_config(scenario: ScenarioCfg, init_states: List, output_filename: str):
    """
    将 ScenarioCfg 对象和 init_states 列表转换为一个简洁、人类可读的 YAML 文件。
    此版本能正确生成带资产引用 (<<: *alias) 的格式。
    """

    def sanitize_value(v):
        if isinstance(v, torch.Tensor): return v.tolist()
        if isinstance(v, PhysicStateType): return v.name
        if isinstance(v, tuple): return list(v)
        if isinstance(v, dict): return {k: sanitize_value(v_new) for k, v_new in v.items()}
        if isinstance(v, list): return [sanitize_value(i) for i in v]
        return v

    # 1. 提取资产 (Assets)
    assets = {}
    if scenario.objects:
        first_obj = scenario.objects[0]
        asset_name = first_obj.name.rsplit('_', 2)[0]
        assets[asset_name] = {
            "scale": list(first_obj.scale),
            "physics": first_obj.physics.name,
            "usd": first_obj.usd_path,
            "urdf": first_obj.urdf_path,
            "mjcf": first_obj.mjcf_path,
        }

    # 2. 构建 'scenario' 部分
    scenario_dict = {
        "robot": scenario.robots[0].name if scenario.robots else "unknown",
        "sim": scenario.sim,
        "num_envs": scenario.num_envs,
        "headless": scenario.headless,
        "try_add_table": scenario.try_add_table,
        "task": "task description placeholder",
        "cameras": [sanitize_value(vars(cam)) for cam in scenario.cameras],
        "objects": [],
        "robots": [r.name for r in scenario.robots],
    }

    # 填充物体信息，并创建特殊的引用结构
    for obj in scenario.objects:
        asset_name_ref = obj.name.rsplit('_', 2)[0]
        obj_entry = {"name": obj.name}
        if asset_name_ref in assets:
            # 创建一个特殊的元组来标记这是一个合并操作
            # (<<, *asset_name_ref)
            obj_entry['<<'] = f"*{asset_name_ref}"
        scenario_dict["objects"].append(obj_entry)

    # 3. 构建 'initial_state' 部分
    initial_state_dict = sanitize_value(init_states[0]) if init_states else None

    # 4. 组合成最终的配置字典
    final_config = {
        "assets": assets,
        "scenario": scenario_dict,
        "initial_state": initial_state_dict,
    }

    # 5. 自定义 Dumper 来处理特殊的合并键
    class CustomDumper(yaml.SafeDumper):
        def represent_mapping(self, tag, mapping, flow_style=None):
            # 查找我们标记的合并键
            if '<<' in mapping and isinstance(mapping['<<'], str) and mapping['<<'].startswith('*'):
                alias = mapping.pop('<<')
                # 创建一个特殊的元组 (key_node, value_node)
                # PyYAML 会正确处理它
                merge_tuple = (yaml.ScalarNode('tag:yaml.org,2002:merge', '<<'),
                               yaml.ScalarNode('tag:yaml.org,2002:str', alias))

                # 将其插入到表示列表的开头
                value = self.represent_data(mapping)
                value.value.insert(0, merge_tuple)
                return value
            return super().represent_mapping(tag, mapping, flow_style)

    # 6. 写入 YAML 文件
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_filename, 'w') as f:
        # 使用自定义的 Dumper
        yaml.dump(final_config, f, Dumper=CustomDumper, sort_keys=False, indent=2)

    log.info(f"✅ The scenario is written to: {output_filename}")

if __name__ == "__main__":
    from openpi_startup.pick_up_middle_bottom import scenario, init_states
    # Example usage of gen_scene_to_config
    # output_filename = os.path.join(rootutils.find_root(), "scene", "config", "scenarios_test_1.yml")
    # gen_scene_to_config(scenario, init_states, output_filename)
    config_name = "scenarios_test_1.yml"  # Replace with your actual config file name
    scene_env, init_obs, init_extras = get_scene_from_config(config_name)
#     log.info(f"Scene environment initialized: {scene_env}")
#     log.info(f"Initial observations: {init_obs}")
#     log.info(f"Initial extras: {init_extras}")
#     # get_scene(config_name)
