import torch
import math


class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, std_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = torch.nn.Parameter(torch.empty(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.empty(out_features))

        self.register_buffer("eps_in", torch.empty(in_features))
        self.register_buffer("eps_out", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma_weight = self.std_init / math.sqrt(self.in_features)
        sigma_bias = self.std_init / math.sqrt(self.out_features)

        self.weight_sigma.data.fill_(sigma_weight)
        self.bias_sigma.data.fill_(sigma_bias)

    @staticmethod
    def f(x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        eps_in = torch.randn(self.in_features)
        eps_out = torch.randn(self.out_features)

        self.eps_in.copy_(self.f(eps_in))
        self.eps_out.copy_(self.f(eps_out))

    def forward(self, x):
        if self.training:
            noise_w = torch.outer(self.eps_out, self.eps_in)
            w = self.weight_mu + self.weight_sigma * noise_w
            b = self.bias_mu + self.bias_sigma * self.eps_out
        else:
            w = self.weight_mu
            b = self.bias_mu
        return torch.nn.functional.linear(x, w, b)
