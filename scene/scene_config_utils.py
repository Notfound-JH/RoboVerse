from __future__ import annotations

import os
import yaml
import importlib
import rootutils
import torch
from typing import List
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
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

def load_class(path: str):
    """Load a class from a string path."""
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def get_scene_from_config(config_name: str):
    """Read the scene configuration from a YAML file."""
    config = read_config(config_name)
    registry_path = "scene/config/registry.yml"
    registry_cfg = read_config(registry_path)
    scenario_data = config['scenario']
    assets = config.get('assets', {})

    # 1. 预处理物体列表，将资产属性合并进去
    processed_objects = []
    for obj_data in scenario_data.get('objects', []):
        # 查找合并键 '<<'
        asset_ref_data = obj_data.pop('<<', None)
        if asset_ref_data:
            # 创建一个新的字典，先放资产属性，再用物体自身属性覆盖
            merged_data = asset_ref_data.copy()
            merged_data.update(obj_data)
            processed_objects.append(merged_data)
        else:
            processed_objects.append(obj_data)

    # 2. 创建 ScenarioCfg
    scenario = ScenarioCfg(
        robots=scenario_data.get('robots', []),
        try_add_table=scenario_data.get('try_add_table', False),
        sim=scenario_data.get('sim', 'isaaclab'),
        headless=scenario_data.get('headless', False),
        num_envs=scenario_data.get('num_envs', 1),
    )
    log.info(f"Scenario configuration loaded from {config_name}: {scenario}")

    # 3. 添加相机
    for cam_data in scenario_data.get('cameras', []):
        if 'clipping_range' in cam_data and isinstance(cam_data['clipping_range'], list):
            cam_data['clipping_range'] = tuple(cam_data['clipping_range'])
    scenario.cameras = [PinholeCameraCfg(**cam_data) for cam_data in scenario_data.get('cameras', [])]

    # 4. 添加物体 (使用我们预处理过的列表)
    object_cfgs = []
    for obj_data in processed_objects:
        # 将字符串形式的 physics 转换为枚举类型
        if 'physics' in obj_data and isinstance(obj_data['physics'], str):
            obj_data['physics'] = PhysicStateType[obj_data['physics']]

        # 根据注册表加载对应的配置类
        if 'type' in obj_data:
            obj_type = obj_data.pop('type')
            if obj_type in registry_cfg:
                class_path = registry_cfg[obj_type]
                obj_class = load_class(class_path)
                object_cfgs.append(obj_class(**obj_data))
            else:
                log.warning(f"Object type '{obj_type}' not found in registry. Using RigidObjCfg as default.")
                object_cfgs.append(RigidObjCfg(**obj_data))
    scenario.objects = object_cfgs

    # 5. 设置初始状态
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
    此版本能正确生成带多种资产引用 (<<: *alias) 且无引号的格式。
    """

    # 辅助函数：将特殊类型转换为YAML基本类型
    def sanitize_value(v):
        if isinstance(v, torch.Tensor): return v.tolist()
        if isinstance(v, PhysicStateType): return v.name
        if isinstance(v, tuple): return list(v)
        # 新增：处理 dataclass 对象
        if hasattr(v, '__dict__'):
             # 过滤掉私有或不需要的属性
            return {k: sanitize_value(v_new) for k, v_new in vars(v).items() if not k.startswith('_')}
        if isinstance(v, dict): return {k: sanitize_value(v_new) for k, v_new in v.items()}
        if isinstance(v, list): return [sanitize_value(i) for i in v]
        return v

    # 1. 提取资产 (Assets) - 支持多种资产
    asset_definitions = {}
    for obj in scenario.objects:
        # 从物体名称推断资产名称，例如 "bbq_sauce_1" -> "bbq_sauce"
        asset_name = obj.name.rsplit('_', 1)[0]
        # 如果是第一次遇到这种类型的资产，则为其创建定义
        if asset_name not in asset_definitions:
            # 将对象配置转换为字典，并移除 'name'，因为名称是实例相关的
            asset_data = sanitize_value(obj)
            asset_data.pop('name', None)
            asset_data['type'] = asset_name  # 添加类型信息
            asset_definitions[asset_name] = asset_data

    # 2. 构建 'scenario' 部分
    scenario_dict = {
        "robots": [r.name for r in scenario.robots],
        "sim": scenario.sim,
        "num_envs": scenario.num_envs,
        "headless": scenario.headless,
        "try_add_table": scenario.try_add_table,
        "task": "task description placeholder",
        "cameras": [sanitize_value(cam) for cam in scenario.cameras],
        "objects": [],
    }

    # 3. 填充物体信息，并引用正确的资产
    for obj in scenario.objects:
        asset_name_ref = obj.name.rsplit('_', 1)[0]
        obj_entry = {"name": obj.name}
        if asset_name_ref in asset_definitions:
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

    # 6. 使用能处理合并键的 CustomDumper
    class CustomDumper(yaml.SafeDumper):
        def represent_mapping(self, tag, mapping, flow_style=None):
            if '<<' in mapping:
                merge_obj = mapping.pop('<<')
                node = super().represent_mapping(tag, mapping, flow_style=flow_style)
                alias_node = self.represent_data(merge_obj)
                merge_key_node = yaml.ScalarNode('tag:yaml.org,2002:merge', '<<')
                node.value.insert(0, (merge_key_node, alias_node))
                return node
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
