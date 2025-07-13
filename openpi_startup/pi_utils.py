import torch
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
