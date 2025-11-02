import torch


def negative_part(x):
    return torch.minimum(x, torch.zeros_like(x))


def positive_part(x):
    return torch.maximum(x, torch.zeros_like(x))


def as_channel_parameters(parameter: torch.Tensor, x: torch.Tensor):
    shape = [1] * x.dim()
    shape[1] = -1
    return parameter.view(*shape)


def device_compatibility_check(model, x: torch.Tensor):
    for p in model.parameters():
        if p.device != x.device or p.dtype != x.dtype:
            p.data = p.data.to(device=x.device, dtype=x.dtype)


def phi_hat(x, a, lam):
    term_pos = torch.maximum(lam - torch.abs(x - a), torch.zeros_like(x))
    term_neg = torch.minimum(torch.abs(x - (a + 2 * lam)) - lam, torch.zeros_like(x))
    return term_pos + term_neg
