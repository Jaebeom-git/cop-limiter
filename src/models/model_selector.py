import torch
from models.GroundFFN import GroundFFN
from models.GroundLink import GroundLink
from models.GroundMamba import GroundMamba, GroundSTMamba
from models.GroundTransformer import GroundTransformer

MODELS = {
    'GroundFFN': GroundFFN,
    'GroundLink': GroundLink,
    'GroundMamba': GroundMamba,
    'GroundSTMamba': GroundSTMamba,
    'GroundTransformer': GroundTransformer,
}

def select_model(config, device: torch.device):
    """
    Selects and returns an instance of the model based on the given model name.

    Args:
        model_name (str): The name of the model to select.
        config: Configuration settings for the model.
        device (torch.device): The device to which the model should be moved.

    Returns:
        torch.nn.Module: An instance of the selected model.
    """
    config.num_dofs = 23
    config.num_joints = 12
    config.root_history_len = 10
    if config.model in MODELS:
        return MODELS[config.model](config, device).to(device)
    else:
        raise ValueError(f"Unsupported model type: {config.model}")

def get_available_models():
    """
    Returns a list of available model names.

    Returns:
        List[str]: A list of available model names.
    """
    return list(MODELS.keys())