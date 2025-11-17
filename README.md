# Core

### Device Management

```py
from homa import cpu, mps, cuda, device

torch.tensor([1, 2, 3, 4, 5]).to(cpu())
torch.tensor([1, 2, 3, 4, 5]).to(cuda())
torch.tensor([1, 2, 3, 4, 5]).to(mps())
torch.tensor([1, 2, 3, 4, 5]).to(device())
```

# Vision

## Resnet

This is the standard ResNet50 module.

You can train the model with a `DataLoader` object.

```py
from homa.vision import Resnet

model = Resnet(num_classes=10, lr=0.001)
for epoch in range(10):
    model.train(train_dataloader)
```

Similarly you can manually take care of decomposition of data from the `DataLoader`.

```py
from homa.vision import Resnet

model = Resnet(num_classes=10, lr=0.001)
for epoch in range(10):
    for x, y in train_dataloader:
        model.train(x, y)
```

## StochasticResnet

This is a ResNet module whose activation functions are replaced from a pool of different activation functions randomly. Read more on the [(paper)](https://www.mdpi.com/1424-8220/22/16/6129).

You can train the model with a `DataLoader` object.

```py
from homa.vision import StochasticResnet

model = StochasticResnet(num_classes=10, lr=0.001)
for epoch in range(10):
    model.train(train_dataloader)
```

Similarly you can manually take care of decomposition of data from the `DataLoader`.

```py
from homa.vision import StochasticResnet

model = StochasticResnet(num_classes=10, lr=0.001)
for epoch in range(10):
    for x, y in train_dataloader:
        model.train(x, y)
```

# Activation Functions

The table below lists every module that subclasses `ActivationFunction`, summarizing the computation performed in `forward`, linking to the implementation, and indicating whether the module exposes learnable parameters.

| Activation                 | Formula                                                                                                       | Notes                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------- | ------------- | -------- | --- | --- | --- | ------------------ | ------------- |
| ADA                        | \( f(x) = \begin{cases} x & x \ge 0 \\ x e^x & x < 0 \end{cases} \)                                           | Learnable: ❌                                    |
| AOAF                       | \( f(x) = \max(0, x - b \cdot a) + c \cdot a \)                                                               | Learnable: ✅ (channel-wise adaptive parameters) |
| AReLU                      | \( f(x) = (1 + \sigma(b)) \cdot \max(0, x) + \text{clamp}(a, 0.01, 0.99) \cdot \min(0, x) \)                  | Learnable: ✅ (global adaptive parameters)       |
| ASiLU                      | \( f(x) = \arctan\Big(x \cdot \sigma(x)\Big) \)                                                               | Learnable: ❌                                    |
| AbsLU                      | \( f(x) = \begin{cases} x & x \ge 0 \\ \alpha                                                                 | x                                                | & x < 0 \end{cases} \)                                     | Learnable: ❌ |
| AdaptiveActivationFunction | —                                                                                                             | Base class, no forward                           | Learnable: ❌                                              |
| BaseDLReLU                 | \( f(z) = \begin{cases} z & z \ge 0 \\ a \cdot b_t \cdot z & z < 0 \end{cases} \)                             | Learnable: ❌                                    |
| CaLU                       | \( f(x) = x \left( \frac{\arctan(x)}{\pi} + 0.5 \right) \)                                                    | Learnable: ❌                                    |
| DLReLU                     | Inherits BaseDLReLU                                                                                           | Learnable: ❌                                    |
| DLU                        | \( f(x) = \begin{cases} x & x \ge 0 \\ \frac{x}{1 - x} & x < 0 \end{cases} \)                                 | Learnable: ❌                                    |
| DPReLU                     | \( f(x) = \begin{cases} a x & x \ge 0 \\ b x & x < 0 \end{cases} \)                                           | Learnable: ✅ (channel-wise adaptive parameters) |
| DRLU                       | \( f(x) = \max(0, x - \alpha) \)                                                                              | Learnable: ❌                                    |
| DerivativeSiLU             | \( f(x) = \sigma(x) \big( 1 + x (1 - \sigma(x)) \big) \)                                                      | Learnable: ❌                                    |
| DiffELU                    | \( f(x) = \begin{cases} x & x \ge 0 \\ a (x e^x - b e^{b x}) & x < 0 \end{cases} \)                           | Learnable: ❌                                    |
| DoubleSiLU                 | \( f(x) = \frac{x}{1 + \exp(-(-x/(1 + e^{-x}))) } \)                                                          | Learnable: ❌                                    |
| DualLine                   | \( f(x) = \begin{cases} a x + m & x \ge 0 \\ b x + m & x < 0 \end{cases} \)                                   | Learnable: ✅ (channel-wise adaptive parameters) |
| EANAF                      | \( f(x) = x \cdot g(h(x)) \)                                                                                  | Learnable: ❌                                    |
| Elliot                     | \( f(x) = 0.5 + \frac{0.5 x}{1 +                                                                              | x                                                | } \)                                                       | Learnable: ❌ |
| ExponentialDLReLU          | Inherits BaseDLReLU                                                                                           | Learnable: ❌                                    |
| ExponentialSwish           | \( f(x) = e^{-x} \cdot \sigma(x) \)                                                                           | Learnable: ❌                                    |
| FReLU                      | \( f(x) = \begin{cases} x + b & x \ge 0 \\ b & x < 0 \end{cases} \)                                           | Learnable: ✅ (channel-wise adaptive parameters) |
| FlattedTSwish              | \( f(x) = \max(0, x) \cdot \sigma(x) + t \)                                                                   | Learnable: ❌                                    |
| GeneralizedSwish           | \( f(x) = x \cdot \sigma(e^{-x}) \)                                                                           | Learnable: ❌                                    |
| Gish                       | \( f(x) = x \cdot \ln\big( 2 - e^{-e^x} \big) \)                                                              | Learnable: ❌                                    |
| IpLU                       | \( f(x) = \begin{cases} x & x \ge 0 \\ \frac{x}{1 +                                                           | x                                                | ^\alpha} & x < 0 \end{cases} \)                            | Learnable: ❌ |
| LaLU                       | \( f(x) = x \cdot \begin{cases} 1 - 0.5 e^{-x} & x \ge 0 \\ 0.5 e^x & x < 0 \end{cases} \)                    | Learnable: ❌                                    |
| LeLeLU                     | \( f(x) = \begin{cases} a x & x \ge 0 \\ 0.01 a x & x < 0 \end{cases} \)                                      | Learnable: ✅ (channel-wise adaptive parameters) |
| LogSigmoid                 | \( f(x) = \ln(\sigma(x)) \)                                                                                   | Learnable: ❌                                    |
| Logish                     | \( f(x) = x \cdot \ln(1 + \sigma(x)) \)                                                                       | Learnable: ❌                                    |
| MSiLU                      | \( f(x) = x \sigma(x) + \frac{1}{4} e^{-x^2 - 1} \)                                                           | Learnable: ❌                                    |
| MaxSig                     | \( f(x) = \max(x, \sigma(x)) \)                                                                               | Learnable: ❌                                    |
| MinSin                     | \( f(x) = \min(x, \sin(x)) \)                                                                                 | Learnable: ❌                                    |
| NLReLU                     | \( f(x) = \ln(1 + \beta \cdot \max(0, x)) \)                                                                  | Learnable: ❌                                    |
| NReLU                      | \( f(x) = \begin{cases} x + a & x \ge 0 \\ 0 & x < 0 \end{cases} \)                                           | Learnable: ❌                                    |
| NoisyReLU                  | Inherits NReLU                                                                                                | Learnable: ❌                                    |
| OAF                        | \( f(x) = \max(0, x) + x \cdot \sigma(x) \)                                                                   | Learnable: ❌                                    |
| PERU                       | \( f(x) = \begin{cases} a x & x \ge 0 \\ a x e^{b x} & x < 0 \end{cases} \)                                   | Learnable: ✅ (channel-wise adaptive parameters) |
| PFLU                       | \( f(x) = x \cdot 0.5 \left( 1 + \frac{x}{\sqrt{1 + x^2}} \right) \)                                          | Learnable: ❌                                    |
| PLAF                       | \( f(x) = \begin{cases} x - \delta & x \ge 1 \\ -x - \delta & x < -1 \\                                       | x                                                | ^d / d & -1 \le x < 1 \end{cases} \), \(\delta = 1 - 1/d\) | Learnable: ❌ |
| Phish                      | \( f(x) = x \cdot \tanh(\text{GELU}(x)) \)                                                                    | Learnable: ❌                                    |
| PiLU                       | \( f(x) = \begin{cases} a x + c (1 - a) & x \ge c \\ b x + c (1 - b) & x < c \end{cases} \)                   | Learnable: ✅ (channel-wise adaptive parameters) |
| PoLU                       | \( f(x) = \begin{cases} x & x \ge 0 \\ (1 - x)^{-\alpha} - 1 & x < 0 \end{cases} \)                           | Learnable: ❌                                    |
| PolyLU                     | \( f(x) = \begin{cases} x & x \ge 0 \\ \frac{1}{1 - x} - 1 & x < 0 \end{cases} \)                             | Learnable: ❌                                    |
| REU                        | \( f(x) = \begin{cases} x & x \ge 0 \\ x e^x & x < 0 \end{cases} \)                                           | Learnable: ❌                                    |
| RReLU                      | \( f(x) = \begin{cases} x & x \ge 0 \\ x / a & x < 0 \end{cases} \), \(a \in [\text{lower}, \text{upper}]\)   | Learnable: ❌                                    |
| RandomizedSlopedReLU       | Inherits SlopedReLU                                                                                           | Learnable: ❌                                    |
| ReCU                       | Inherits RePU                                                                                                 | Learnable: ❌                                    |
| RePU                       | \( f(x) = \max(0, x^\alpha) \)                                                                                | Learnable: ❌                                    |
| ReQU                       | Inherits RePU                                                                                                 | Learnable: ❌                                    |
| ReSP                       | \( f(x) = \begin{cases} \alpha x + \ln 2 & x \ge 0 \\ \ln(1 + e^x) & x < 0 \end{cases} \)                     | Learnable: ❌                                    |
| ReSech                     | \( f(x) = x \cdot \text{sech}(x) \)                                                                           | Learnable: ❌                                    |
| SGELU                      | \( f(x) = \alpha x \cdot \text{erf}\left(\frac{x}{\sqrt{2}}\right) \)                                         | Learnable: ❌                                    |
| SaRa                       | \( f(x) = \begin{cases} x & x \ge 0 \\ x / (1 + \alpha e^{-\beta x}) & x < 0 \end{cases} \)                   | Learnable: ❌                                    |
| Serf                       | \( f(x) = x \cdot \text{erf}(\ln(1 + e^x)) \)                                                                 | Learnable: ❌                                    |
| ShiLU                      | \( f(x) = a \cdot \max(0, x) + b \)                                                                           | Learnable: ✅ (channel-wise adaptive parameters) |
| ShiftedReLU                | \( f(x) = \max(x, -1) \)                                                                                      | Learnable: ❌                                    |
| SiELU                      | \( f(x) = x \cdot \sigma(2 \sqrt{2 / \pi} \cdot (x + 0.044715 x^3)) \)                                        | Learnable: ❌                                    |
| SigLU                      | \( f(x) = \begin{cases} x & x \ge 0 \\ \frac{1 - e^{-2x}}{1 + e^{-2x}} & x < 0 \end{cases} \)                 | Learnable: ❌                                    |
| SigmoidDerivative          | \( f(x) = e^{-x} \cdot \sigma(x)^2 \)                                                                         | Learnable: ❌                                    |
| SinSig                     | \( f(x) = x \cdot \sin\left(\frac{\pi}{2} \sigma(x)\right) \)                                                 | Learnable: ❌                                    |
| SineReLU                   | \( f(x) = \begin{cases} x & x \ge 0 \\ \epsilon (\sin x - \cos x) & x < 0 \end{cases} \)                      | Learnable: ❌                                    |
| SlopedReLU                 | \( f(x) = \begin{cases} \alpha x & x \ge 0 \\ 0 & x < 0 \end{cases} \)                                        | Learnable: ❌                                    |
| Smish                      | \( f(x) = x \cdot \tanh(\ln(1 + \sigma(x))) \)                                                                | Learnable: ❌                                    |
| SoftModulusQ               | \( f(x) = \begin{cases} x^2 (2 -                                                                              | x                                                | ) &                                                        | x             | \le 1 \\ | x   | &   | x   | > 1 \end{cases} \) | Learnable: ❌ |
| SoftModulusT               | \( f(x) = x \cdot \tanh(x / \alpha) \)                                                                        | Learnable: ❌                                    |
| SoftsignRReLU              | \( f(x) = \begin{cases} \frac{1}{(1 + x)^2} + x & x \ge 0 \\ \frac{1}{(1 + x)^2} + a x & x < 0 \end{cases} \) | Learnable: ❌                                    |
| StarReLU                   | \( f(x) = a (\max(0, x))^2 + b \)                                                                             | Learnable: ✅ (channel-wise adaptive parameters) |
| Suish                      | \( f(x) = \max(x, x e^{-                                                                                      | x                                                | }) \)                                                      | Learnable: ❌ |
| TBSReLU                    | \( f(x) = x \cdot \tanh\left(\frac{1 - e^{-x}}{1 + e^{-x}}\right) \)                                          | Learnable: ❌                                    |
| TSReLU                     | \( f(x) = x \cdot \tanh(\sigma(x)) \)                                                                         | Learnable: ❌                                    |
| TSiLU                      | \( f(x) = \frac{e^\alpha - e^{-\alpha}}{e^\alpha + e^\alpha}, \ \alpha = \frac{x}{1 + e^{-x}} \)              | Learnable: ❌                                    |
| TangentBipolarSigmoidReLU  | Inherits TBSReLU                                                                                              | Learnable: ❌                                    |
| TangentSigmoidReLU         | Inherits TSReLU                                                                                               | Learnable: ❌                                    |
| TanhExp                    | \( f(x) = x \cdot \tanh(e^x) \)                                                                               | Learnable: ❌                                    |
| TeLU                       | \( f(x) = x \cdot \tanh(e^x) \)                                                                               | Learnable: ❌                                    |
| ThLU                       | \( f(x) = \begin{cases} x & x \ge 0 \\ \tanh(x/2) & x < 0 \end{cases} \)                                      | Learnable: ❌                                    |
| TripleStateSwish           | \( f(x) = x \cdot a \cdot (a + b + c),\ a = \sigma(x),\ b = \sigma(x - \alpha),\ c = \sigma(x - \beta) \)     | Learnable: ❌                                    |
| mReLU                      | \( f(x) = \min(\max(0, 1 - x), \max(0, 1 + x)) \)                                                             | Learnable: ❌                                    |
