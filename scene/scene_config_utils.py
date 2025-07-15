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
    with open(config_name, "r") as f:
        config = yaml.safe_load(f)
    import pprint
    pprint.pprint(config)
    return config

def get_scene_from_config(config_name: str):
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
    # 【关键修改】在创建 PinholeCameraCfg 之前，将 clipping_range 从列表转换为元组
    for cam_data in scenario_data.get('cameras', []):
        if 'clipping_range' in cam_data and isinstance(cam_data['clipping_range'], list):
            cam_data['clipping_range'] = tuple(cam_data['clipping_range'])

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
    return scenario, init_states

def gen_scene_to_config(scenario: ScenarioCfg, init_states: List, output_filename: str):
    """
    将 ScenarioCfg 对象和 init_states 列表转换为一个简洁、人类可读的 YAML 文件。
    此版本能正确生成带资产引用 (<<: *alias) 且无引号的格式。
    """

    # 辅助函数：将特殊类型转换为YAML基本类型
    def sanitize_value(v):
        if isinstance(v, torch.Tensor): return v.tolist()
        if isinstance(v, PhysicStateType): return v.name
        if isinstance(v, tuple): return list(v)
        if isinstance(v, dict): return {k: sanitize_value(v_new) for k, v_new in v.items()}
        if isinstance(v, list): return [sanitize_value(i) for i in v]
        return v

    # 1. 提取资产 (Assets)
    asset_definitions = {}
    for object in scenario.objects:
        asset_name = object.name.rsplit('_', 1)[0]
        # 这个 asset_data 对象将在多处被引用
        asset_data = {
            "scale": list(object.scale),
            "physics": object.physics.name,
            "usd_path": object.usd_path,
            "urdf_path": object.urdf_path,
            "mjcf_path": object.mjcf_path,
        }
        asset_definitions[asset_name] = asset_data

    # 2. 构建 'scenario' 部分
    scenario_dict = {
        "robots": [r.name for r in scenario.robots],
        "sim": scenario.sim,
        "num_envs": scenario.num_envs,
        "headless": scenario.headless,
        "try_add_table": scenario.try_add_table,
        "task": "task description placeholder",
        "cameras": [sanitize_value(vars(cam)) for cam in scenario.cameras],
        "objects": [],
    }

    # 3. 填充物体信息，并引用共享的资产对象
    for obj in scenario.objects:
        asset_name_ref = obj.name.rsplit('_', 1)[0]
        obj_entry = {"name": obj.name}
        if asset_name_ref in asset_definitions:
            # 关键：将共享的资产字典对象本身赋值给 '<<' 键
            obj_entry['<<'] = asset_definitions[asset_name_ref]
        scenario_dict["objects"].append(obj_entry)

    # 4. 构建 'initial_state' 部分
    initial_state_dict = sanitize_value(init_states[0]) if init_states else None

    # 5. 组合成最终的配置字典
    final_config = {
        "assets": asset_definitions,
        "scenario": scenario_dict,
        "initial_state": initial_state_dict,
    }

    # 6. 创建能正确处理合并键的 CustomDumper
    class CustomDumper(yaml.SafeDumper):
        def represent_mapping(self, tag, mapping, flow_style=None):
            # 检查字典中是否有我们定义的合并键
            if '<<' in mapping:
                # 将 '<<' 键值对从字典中弹出
                merge_obj = mapping.pop('<<')

                # 首先，正常处理字典中剩余的普通键值对 (如 'name')
                node = super().represent_mapping(tag, mapping, flow_style=flow_style)

                # 然后，单独处理我们弹出的合并对象
                # PyYAML 会自动检测到这是一个已见过的对象，并生成别名节点
                alias_node = self.represent_data(merge_obj)

                # 创建一个不带引号的合并键 '<<' 节点
                merge_key_node = yaml.ScalarNode('tag:yaml.org,2002:merge', '<<')

                # 将 (合并键, 别名) 节点对插入到节点列表
                node.value.insert(1, (merge_key_node, alias_node))

                return node

            # 如果没有 '<<' 键，则正常处理
            return super().represent_mapping(tag, mapping, flow_style=flow_style)

    # 7. 写入 YAML 文件
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_filename, 'w') as f:
        yaml.dump(final_config, f, Dumper=CustomDumper, sort_keys=False, indent=2)

    log.info(f"✅ The scenario is written to: {output_filename}")

if __name__ == "__main__":
    from openpi_startup.pick_up_middle_bottom import scenario, init_states
    # Example usage of gen_scene_to_config
    # output_filename = "scenarios_test_1.yml"
    # gen_scene_to_config(scenario, init_states, output_filename)
    config_path = "scene/config/scenarios_test.yml"  # Replace with your actual config file name
    scene_env, init_obs, init_extras = get_scene_from_config(config_path)
#     log.info(f"Scene environment initialized: {scene_env}")
#     log.info(f"Initial observations: {init_obs}")
#     log.info(f"Initial extras: {init_extras}")
#     # get_scene(config_name)
