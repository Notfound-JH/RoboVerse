import collections
import dataclasses
import logging
import pathlib
import imageio
import numpy as np
import tqdm
import tyro
# 2. 导入 openpi 客户端
from openpi_client import websocket_client_policy as _websocket_client_policy

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

def eval_my_env(args: Args) -> None:
    # 设置随机种子
    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    env.seed(args.seed)
    task_description = env.get_task_description() # 假设您的环境有这样一个方法

    # 连接到策略服务器
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # 开始评估
    total_episodes, total_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task: {task_description}"):
        logging.info(f"\nTask: {task_description}")
        # 重置环境
        # 6. 这里的 `obs` 是您环境返回的原始观察字典
        obs = env.reset()
        action_plan = collections.deque()
        # 设置
        t = 0
        replay_images = [] # 用于录制视频的图像帧
        logging.info(f"Starting episode {episode_idx + 1}...")
        while t < args.max_steps + args.num_steps_wait:
            try:
                # 等待模拟器稳定
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(ROBOVERSE_DUMMY_ACTION)
                    t += 1
                    continue

                # 7. 录制视频帧 (从原始观察中获取)
                #    这里的键名 'main_camera_rgb' 需要匹配您环境的输出
                replay_images.append(obs["camera0"])

                if not action_plan:
                    # 8. 准备发送给模型的字典 (关键改动)
                    #    这里的键名需要和您在 `my_env_policy.py` 的 `MyEnvInputs` 中
                    #    期望接收的键名完全一致！
                    #    您不再需要在这里进行图像旋转、缩放或状态拼接。
                    element = {
                        "observation/image": obs["camera0"],
                        "observation/wrist_image": obs["camera1"],
                        "observation/state": obs["robot_state"],
                        "language_instruction": str(task_description),
                    }
                    # 查询模型以获取动作
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()

                # 在环境中执行动作
                obs, reward, done, info = env.step(action) # 无需 .tolist() 如果 action 已经是 list/numpy
                if done:
                    total_successes += 1
                    break
                t += 1

            except Exception as e:
                logging.error(f"Caught exception: {e}", exc_info=True)
                break

        total_episodes += 1
        is_success = done # 假设 done=True 代表成功

        # 保存回放视频
        suffix = "success" if is_success else "failure"
        task_segment = args.env_id.replace("/", "_")
        video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4"
        imageio.mimwrite(video_path, replay_images, fps=15, quality=8)
        logging.info(f"Saved video to {video_path}")

        # 打印日志
        logging.info(f"Success: {is_success}")
        logging.info(f"Success rate so far: {total_successes / total_episodes * 100:.1f}% ({total_successes}/{total_episodes})")

    logging.info(f"Final success rate: {total_successes / total_episodes * 100:.1f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_my_env)
