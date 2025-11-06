import torch


def get_model_device(model: torch.nn.Module):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    return device
