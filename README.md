# Core

## Device Management

This section demonstrates how to easily select computation devices when using PyTorch. The `homa` helpers provide consistent interfaces for CPU, CUDA-enabled GPUs, and Apple Silicon MPS.

- `cpu()`: Forces tensors or models onto the CPU.
- `cuda()`: Moves tensors or models onto a CUDA GPU (if available). Commonly used in high‑performance training.
- `mps()`: Uses Apple's Metal Performance Shaders backend on macOS.
- `get_device()`: Automatically infers the best available device in the order: CUDA → MPS → CPU.

```py
from homa import cpu, mps, cuda, get_device

# explicitly selecting devices
torch.tensor([1, 2, 3, 4, 5]).to(cpu())
torch.tensor([1, 2, 3, 4, 5]).to(cuda())
torch.tensor([1, 2, 3, 4, 5]).to(mps())

# automatic device selection
torch.tensor([1, 2, 3, 4, 5]).to(get_device())
```

This design mirrors common best practices in deep learning workflows, promoting device‑agnostic code.

## Loading Settings

`homa.settings` allows you to attach a `settings.json` file to your project and access its values directly in your code. This is useful for hyperparameters, configuration management, or experiment logging.

Example `settings.json`:

```json
{
  "epochs": 100,
  "learning_rate": 0.001
}
```

Loading settings in Python:

```py
from homa import settings

for epoch in range(settings("epochs")):
    pass
```

The helper reads and caches the JSON content, providing dictionary‑like access without requiring boilerplate file‑loading logic.

# Vision

## Resnet

`homa.vision.Resnet` implements a standard **ResNet‑50** architecture, commonly used in image classification tasks. This class bundles the model, optimizer, and training loop helpers for fast prototyping.

You can train the model directly using a PyTorch `DataLoader`:

```py
from homa.vision import Resnet

model = Resnet(num_classes=10, lr=0.001)
for epoch in range(10):
    model.train(train_dataloader)
```

Alternatively, you may manually unpack the DataLoader and pass data batches yourself:

```py
from homa.vision import Resnet

model = Resnet(num_classes=10, lr=0.001)
for epoch in range(10):
    for x, y in train_dataloader:
        model.train(x, y)
```

This interface is influenced by modern PyTorch training utilities and mirrors patterns seen in high‑level frameworks while keeping full transparency over the training loop.

# Loss Functions

## Logit Normalization

**LogitNorm** is a modified cross‑entropy‑style loss that normalizes logits before computing the loss. This technique was introduced to improve calibration, robustness, and especially performance in **ensembling** scenarios where varied model outputs can lead to instability.

Typical benefits of LogitNorm include:

- more stable gradients
- improved probabilistic calibration
- robustness to logit scaling differences across models

```py
from homa.loss import LogitNorm

criterion = LogitNorm()
```

Logit normalization is related to works studying the effect of logit scaling on generalization and calibration in deep networks.

# Activation Functions

The table below lists every module that subclasses `ActivationFunction`, summarizing the computation performed in `forward`, linking to the implementation, and indicating whether the module exposes learnable parameters.

<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Activation</th>
      <th>Formula (plain text)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>ADA</td><td>f(x) = x if x ≥ 0, else x * exp(x)</td></tr>
    <tr><td>2</td><td>AOAF</td><td>f(x) = max(0, x − b · a) + c · a</td></tr>
    <tr><td>3</td><td>AReLU</td><td>f(x) = (1 + σ(b)) * max(0, x) + clamp(a, 0.01, 0.99) * min(0, x)</td></tr>
    <tr><td>4</td><td>ASiLU</td><td>f(x) = arctan(x * σ(x))</td></tr>
    <tr><td>5</td><td>AbsLU</td><td>f(x) = x if x ≥ 0, else α * |x|</td></tr>
    <tr><td>6</td><td>BaseDLReLU</td><td>f(x) = x if x ≥ 0, else a * bₜ * x</td></tr>
    <tr><td>7</td><td>CaLU</td><td>f(x) = x * (arctan(x) / π + 0.5)</td></tr>
    <tr><td>8</td><td>DLReLU</td><td>Inherits BaseDLReLU</td></tr>
    <tr><td>9</td><td>DLU</td><td>f(x) = x if x ≥ 0, else x / (1 − x)</td></tr>
    <tr><td>10</td><td>DPReLU</td><td>f(x) = a x if x ≥ 0, else b x</td></tr>
    <tr><td>11</td><td>DRLU</td><td>f(x) = max(0, x − α)</td></tr>
    <tr><td>12</td><td>DerivativeSiLU</td><td>f(x) = σ(x) * (1 + x (1 − σ(x)))</td></tr>
    <tr><td>13</td><td>DiffELU</td><td>f(x) = x if x ≥ 0, else a * (x eˣ − b e^(b x))</td></tr>
    <tr><td>14</td><td>DoubleSiLU</td><td>f(x) = x / (1 + exp(−(−x / (1 + e^(−x)))))</td></tr>
    <tr><td>15</td><td>DualLine</td><td>f(x) = a x + m if x ≥ 0, else b x + m</td></tr>
    <tr><td>16</td><td>EANAF</td><td>f(x) = x · g(h(x))</td></tr>
    <tr><td>17</td><td>Elliot</td><td>f(x) = 0.5 + (0.5 x)/(1 + |x|)</td></tr>
    <tr><td>18</td><td>ExponentialDLReLU</td><td>Inherits BaseDLReLU</td></tr>
    <tr><td>19</td><td>ExponentialSwish</td><td>f(x) = exp(−x) · σ(x)</td></tr>
    <tr><td>20</td><td>FReLU</td><td>f(x) = x + b if x ≥ 0, else b</td></tr>
    <tr><td>21</td><td>FlattedTSwish</td><td>f(x) = max(0, x) · σ(x) + t</td></tr>
    <tr><td>22</td><td>GeneralizedSwish</td><td>f(x) = x · σ(exp(−x))</td></tr>
    <tr><td>23</td><td>Gish</td><td>f(x) = x · ln(2 − exp(−exp(x)))</td></tr>
    <tr><td>24</td><td>IpLU</td><td>f(x) = x if x ≥ 0, else x / (1 + |x|^α)</td></tr>
    <tr><td>25</td><td>LaLU</td><td>f(x) = x · (1 − 0.5 e^(−x)) if x ≥ 0, else x · (0.5 e^x)</td></tr>
    <tr><td>26</td><td>LeLeLU</td><td>f(x) = a x if x ≥ 0, else 0.01 a x</td></tr>
    <tr><td>27</td><td>LogSigmoid</td><td>f(x) = ln(σ(x))</td></tr>
    <tr><td>28</td><td>Logish</td><td>f(x) = x · ln(1 + σ(x))</td></tr>
    <tr><td>29</td><td>MSiLU</td><td>f(x) = x σ(x) + 1/4 · e^(−x² − 1)</td></tr>
    <tr><td>30</td><td>MaxSig</td><td>f(x) = max(x, σ(x))</td></tr>
    <tr><td>31</td><td>MinSin</td><td>f(x) = min(x, sin(x))</td></tr>
    <tr><td>32</td><td>NLReLU</td><td>f(x) = ln(1 + β · max(0, x))</td></tr>
    <tr><td>33</td><td>NReLU</td><td>f(x) = x + a if x ≥ 0, else 0</td></tr>
    <tr><td>34</td><td>NoisyReLU</td><td>Inherits NReLU</td></tr>
    <tr><td>35</td><td>OAF</td><td>f(x) = max(0, x) + x · σ(x)</td></tr>
    <tr><td>36</td><td>PERU</td><td>f(x) = a x if x ≥ 0, else a x · e^(b x)</td></tr>
    <tr><td>37</td><td>PFLU</td><td>f(x) = x · 0.5 · (1 + x / √(1 + x²))</td></tr>
    <tr><td>38</td><td>PLAF</td><td>f(x) = x − δ if x ≥ 1; = −x − δ if x < −1; = |x|^d / d otherwise; δ = 1 − 1/d</td></tr>
    <tr><td>39</td><td>Phish</td><td>f(x) = x · tanh(GELU(x))</td></tr>
    <tr><td>40</td><td>PiLU</td><td>f(x) = a x + c (1 − a) if x ≥ c, else b x + c (1 − b)</td></tr>
    <tr><td>41</td><td>PoLU</td><td>f(x) = x if x ≥ 0, else (1 − x)^(−α) − 1</td></tr>
    <tr><td>42</td><td>PolyLU</td><td>f(x) = x if x ≥ 0, else 1/(1 − x) − 1</td></tr>
    <tr><td>43</td><td>REU</td><td>f(x) = x if x ≥ 0, else x · exp(x)</td></tr>
    <tr><td>44</td><td>RReLU</td><td>f(x) = x if x ≥ 0, else x / a, where a ∈ [lower, upper]</td></tr>
    <tr><td>45</td><td>RandomizedSlopedReLU</td><td>Inherits SlopedReLU</td></tr>
    <tr><td>46</td><td>ReCU</td><td>Inherits RePU</td></tr>
    <tr><td>47</td><td>RePU</td><td>f(x) = max(0, x^α)</td></tr>
    <tr><td>48</td><td>ReQU</td><td>Inherits RePU</td></tr>
    <tr><td>49</td><td>ReSP</td><td>f(x) = α x + ln 2 if x ≥ 0, else ln(1 + e^x)</td></tr>
    <tr><td>50</td><td>ReSech</td><td>f(x) = x · sech(x)</td></tr>
    <tr><td>51</td><td>SGELU</td><td>f(x) = α x · erf(x / √2)</td></tr>
    <tr><td>52</td><td>SaRa</td><td>f(x) = x if x ≥ 0, else x / (1 + α e^(−β x))</td></tr>
    <tr><td>53</td><td>Serf</td><td>f(x) = x · erf(ln(1 + e^x))</td></tr>
    <tr><td>54</td><td>ShiLU</td><td>f(x) = a · max(0, x) + b</td></tr>
    <tr><td>55</td><td>ShiftedReLU</td><td>f(x) = max(x, −1)</td></tr>
    <tr><td>56</td><td>SiELU</td><td>f(x) = x · σ(2 √(2/π) · (x + 0.044715 x³))</td></tr>
    <tr><td>57</td><td>SigLU</td><td>f(x) = x if x ≥ 0, else (1 − e^(−2x)) / (1 + e^(−2x))</td></tr>
    <tr><td>58</td><td>SigmoidDerivative</td><td>f(x) = e^(−x) · σ(x)²</td></tr>
    <tr><td>59</td><td>SinSig</td><td>f(x) = x · sin((π/2) · σ(x))</td></tr>
    <tr><td>60</td><td>SineReLU</td><td>f(x) = x if x ≥ 0, else ε · (sin x − cos x)</td></tr>
    <tr><td>61</td><td>SlopedReLU</td><td>f(x) = α x if x ≥ 0, else 0</td></tr>
    <tr><td>62</td><td>Smish</td><td>f(x) = x · tanh(ln(1 + σ(x)))</td></tr>
    <tr><td>63</td><td>SoftModulusQ</td><td>f(x) = x² (2 − |x|) if |x| ≤ 1, else |x|</td></tr>
    <tr><td>64</td><td>SoftModulusT</td><td>f(x) = x · tanh(x / α)</td></tr>
    <tr><td>65</td><td>SoftsignRReLU</td><td>f(x) = 1/(1 + x)² + x if x ≥ 0, else 1/(1 + x)² + a x</td></tr>
    <tr><td>66</td><td>StarReLU</td><td>f(x) = a · (max(0, x))² + b</td></tr>
    <tr><td>67</td><td>Suish</td><td>f(x) = max(x, x · exp(−|x|))</td></tr>
    <tr><td>68</td><td>TBSReLU</td><td>f(x) = x · tanh((1 − e^(−x)) / (1 + e^(−x)))</td></tr>
    <tr><td>69</td><td>TSReLU</td><td>f(x) = x · tanh(σ(x))</td></tr>
    <tr><td>70</td><td>TSiLU</td><td>Let α = x / (1 + e^(−x)), then f(x) = (e^α − e^(−α)) / (e^α + e^(−α))</td></tr>
    <tr><td>71</td><td>TangentBipolarSigmoidReLU</td><td>Inherits TBSReLU</td></tr>
    <tr><td>72</td><td>TangentSigmoidReLU</td><td>Inherits TSReLU</td></tr>
    <tr><td>73</td><td>TanhExp</td><td>f(x) = x · tanh(e^x)</td></tr>
    <tr><td>74</td><td>TeLU</td><td>f(x) = x · tanh(e^x)</td></tr>
    <tr><td>75</td><td>ThLU</td><td>f(x) = x if x ≥ 0, else tanh(x / 2)</td></tr>
    <tr><td>76</td><td>TripleStateSwish</td><td>Let a = σ(x), b = σ(x − α), c = σ(x − β); f(x) = x · a · (a + b + c)</td></tr>
    <tr><td>77</td><td>mReLU</td><td>f(x) = min(max(0, 1 − x), max(0, 1 + x))</td></tr>
  </tbody>
</table>
