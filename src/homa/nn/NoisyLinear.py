import torch
import math


class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = torch.nn.Parameter(
            torch.tensor(out_features, in_features, dtype=torch.float32)
        )
        self.weight_sigma = torch.nn.Parameter(torch.tensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon",
            torch.tensor(out_features, in_features, dtype=torch.float32),
        )
        self.bias_mu = torch.nn.Parameter(
            torch.tensor(out_features, dtype=torch.float32)
        )
        self.bias_sigma = torch.nn.Parameter(
            torch.tensor(out_features, dtype=torch.float32)
        )
        self.register_buffer(
            "bias_epsilon", torch.tensor(out_features, dtype=torch.float32)
        )
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        # this should be called before every training loop iteration
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(input, weight, bias)
