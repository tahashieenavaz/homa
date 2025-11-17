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

<table>
  <thead>
    <tr>
      <th>Activation</th>
      <th>Formula</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>ADA</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ x e^x & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>AOAF</td><td>\( f(x) = \max(0, x - b \cdot a) + c \cdot a \)</td><td>Learnable: ✅ (channel-wise adaptive parameters)</td></tr>
    <tr><td>AReLU</td><td>\( f(x) = (1 + \sigma(b)) \cdot \max(0, x) + \text{clamp}(a, 0.01, 0.99) \cdot \min(0, x) \)</td><td>Learnable: ✅ (global adaptive parameters)</td></tr>
    <tr><td>ASiLU</td><td>\( f(x) = \arctan(x \cdot \sigma(x)) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>AbsLU</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ \alpha |x| & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>AdaptiveActivationFunction</td><td>—</td><td>Base class, no forward | Learnable: ❌</td></tr>
    <tr><td>BaseDLReLU</td><td>\( f(z) = \begin{cases} z & z \ge 0 \\ a \cdot b_t \cdot z & z < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>CaLU</td><td>\( f(x) = x \left( \frac{\arctan(x)}{\pi} + 0.5 \right) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>DLReLU</td><td>Inherits BaseDLReLU</td><td>Learnable: ❌</td></tr>
    <tr><td>DLU</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ \frac{x}{1 - x} & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>DPReLU</td><td>\( f(x) = \begin{cases} a x & x \ge 0 \\ b x & x < 0 \end{cases} \)</td><td>Learnable: ✅ (channel-wise adaptive parameters)</td></tr>
    <tr><td>DRLU</td><td>\( f(x) = \max(0, x - \alpha) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>DerivativeSiLU</td><td>\( f(x) = \sigma(x) \big( 1 + x (1 - \sigma(x)) \big) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>DiffELU</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ a (x e^x - b e^{b x}) & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>DoubleSiLU</td><td>\( f(x) = \frac{x}{1 + \exp(-(-x/(1 + e^{-x}))) } \)</td><td>Learnable: ❌</td></tr>
    <tr><td>DualLine</td><td>\( f(x) = \begin{cases} a x + m & x \ge 0 \\ b x + m & x < 0 \end{cases} \)</td><td>Learnable: ✅ (channel-wise adaptive parameters)</td></tr>
    <tr><td>EANAF</td><td>\( f(x) = x \cdot g(h(x)) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>Elliot</td><td>\( f(x) = 0.5 + \frac{0.5 x}{1 + |x|} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>ExponentialDLReLU</td><td>Inherits BaseDLReLU</td><td>Learnable: ❌</td></tr>
    <tr><td>ExponentialSwish</td><td>\( f(x) = e^{-x} \cdot \sigma(x) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>FReLU</td><td>\( f(x) = \begin{cases} x + b & x \ge 0 \\ b & x < 0 \end{cases} \)</td><td>Learnable: ✅ (channel-wise adaptive parameters)</td></tr>
    <tr><td>FlattedTSwish</td><td>\( f(x) = \max(0, x) \cdot \sigma(x) + t \)</td><td>Learnable: ❌</td></tr>
    <tr><td>GeneralizedSwish</td><td>\( f(x) = x \cdot \sigma(e^{-x}) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>Gish</td><td>\( f(x) = x \cdot \ln(2 - e^{-e^x}) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>IpLU</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ \frac{x}{1 + |x|^\alpha} & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>LaLU</td><td>\( f(x) = x \cdot \begin{cases} 1 - 0.5 e^{-x} & x \ge 0 \\ 0.5 e^x & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>LeLeLU</td><td>\( f(x) = \begin{cases} a x & x \ge 0 \\ 0.01 a x & x < 0 \end{cases} \)</td><td>Learnable: ✅ (channel-wise adaptive parameters)</td></tr>
    <tr><td>LogSigmoid</td><td>\( f(x) = \ln(\sigma(x)) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>Logish</td><td>\( f(x) = x \cdot \ln(1 + \sigma(x)) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>MSiLU</td><td>\( f(x) = x \sigma(x) + \frac{1}{4} e^{-x^2 - 1} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>MaxSig</td><td>\( f(x) = \max(x, \sigma(x)) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>MinSin</td><td>\( f(x) = \min(x, \sin(x)) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>NLReLU</td><td>\( f(x) = \ln(1 + \beta \cdot \max(0, x)) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>NReLU</td><td>\( f(x) = \begin{cases} x + a & x \ge 0 \\ 0 & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>NoisyReLU</td><td>Inherits NReLU</td><td>Learnable: ❌</td></tr>
    <tr><td>OAF</td><td>\( f(x) = \max(0, x) + x \cdot \sigma(x) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>PERU</td><td>\( f(x) = \begin{cases} a x & x \ge 0 \\ a x e^{b x} & x < 0 \end{cases} \)</td><td>Learnable: ✅ (channel-wise adaptive parameters)</td></tr>
    <tr><td>PFLU</td><td>\( f(x) = x \cdot 0.5 \left( 1 + \frac{x}{\sqrt{1 + x^2}} \right) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>PLAF</td><td>\( f(x) = \begin{cases} x - \delta & x \ge 1 \\ -x - \delta & x < -1 \\ |x|^d / d & -1 \le x < 1 \end{cases} \), \(\delta = 1 - 1/d\)</td><td>Learnable: ❌</td></tr>
    <tr><td>Phish</td><td>\( f(x) = x \cdot \tanh(\text{GELU}(x)) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>PiLU</td><td>\( f(x) = \begin{cases} a x + c (1 - a) & x \ge c \\ b x + c (1 - b) & x < c \end{cases} \)</td><td>Learnable: ✅ (channel-wise adaptive parameters)</td></tr>
    <tr><td>PoLU</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ (1 - x)^{-\alpha} - 1 & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>PolyLU</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ \frac{1}{1 - x} - 1 & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>REU</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ x e^x & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>RReLU</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ x / a & x < 0 \end{cases} \), \(a \in [\text{lower}, \text{upper}]\)</td><td>Learnable: ❌</td></tr>
    <tr><td>RandomizedSlopedReLU</td><td>Inherits SlopedReLU</td><td>Learnable: ❌</td></tr>
    <tr><td>ReCU</td><td>Inherits RePU</td><td>Learnable: ❌</td></tr>
    <tr><td>RePU</td><td>\( f(x) = \max(0, x^\alpha) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>ReQU</td><td>Inherits RePU</td><td>Learnable: ❌</td></tr>
    <tr><td>ReSP</td><td>\( f(x) = \begin{cases} \alpha x + \ln 2 & x \ge 0 \\ \ln(1 + e^x) & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>ReSech</td><td>\( f(x) = x \cdot \text{sech}(x) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>SGELU</td><td>\( f(x) = \alpha x \cdot \text{erf}\left(\frac{x}{\sqrt{2}}\right) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>SaRa</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ x / (1 + \alpha e^{-\beta x}) & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>Serf</td><td>\( f(x) = x \cdot \text{erf}(\ln(1 + e^x)) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>ShiLU</td><td>\( f(x) = a \cdot \max(0, x) + b \)</td><td>Learnable: ✅ (channel-wise adaptive parameters)</td></tr>
    <tr><td>ShiftedReLU</td><td>\( f(x) = \max(x, -1) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>SiELU</td><td>\( f(x) = x \cdot \sigma(2 \sqrt{2 / \pi} (x + 0.044715 x^3)) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>SigLU</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ \frac{1 - e^{-2x}}{1 + e^{-2x}} & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>SigmoidDerivative</td><td>\( f(x) = e^{-x} \cdot \sigma(x)^2 \)</td><td>Learnable: ❌</td></tr>
    <tr><td>SinSig</td><td>\( f(x) = x \cdot \sin\left(\frac{\pi}{2} \sigma(x)\right) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>SineReLU</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ \epsilon (\sin x - \cos x) & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>SlopedReLU</td><td>\( f(x) = \begin{cases} \alpha x & x \ge 0 \\ 0 & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>Smish</td><td>\( f(x) = x \cdot \tanh(\ln(1 + \sigma(x))) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>SoftModulusQ</td><td>\( f(x) = \begin{cases} x^2 (2 - |x|) & |x| \le 1 \\ |x| & |x| > 1 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>SoftModulusT</td><td>\( f(x) = x \cdot \tanh(x / \alpha) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>SoftsignRReLU</td><td>\( f(x) = \begin{cases} \frac{1}{(1 + x)^2} + x & x \ge 0 \\ \frac{1}{(1 + x)^2} + a x & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>StarReLU</td><td>\( f(x) = a (\max(0, x))^2 + b \)</td><td>Learnable: ✅ (channel-wise adaptive parameters)</td></tr>
    <tr><td>Suish</td><td>\( f(x) = \max(x, x e^{-|x|}) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>TBSReLU</td><td>\( f(x) = x \cdot \tanh\left(\frac{1 - e^{-x}}{1 + e^{-x}}\right) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>TSReLU</td><td>\( f(x) = x \cdot \tanh(\sigma(x)) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>TSiLU</td><td>\( f(x) = \frac{e^\alpha - e^{-\alpha}}{e^\alpha + e^\alpha}, \alpha = x / (1 + e^{-x}) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>TangentBipolarSigmoidReLU</td><td>Inherits TBSReLU</td><td>Learnable: ❌</td></tr>
    <tr><td>TangentSigmoidReLU</td><td>Inherits TSReLU</td><td>Learnable: ❌</td></tr>
    <tr><td>TanhExp</td><td>\( f(x) = x \cdot \tanh(e^x) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>TeLU</td><td>\( f(x) = x \cdot \tanh(e^x) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>ThLU</td><td>\( f(x) = \begin{cases} x & x \ge 0 \\ \tanh(x/2) & x < 0 \end{cases} \)</td><td>Learnable: ❌</td></tr>
    <tr><td>TripleStateSwish</td><td>\( f(x) = x a (a + b + c), a = \sigma(x), b = \sigma(x-\alpha), c = \sigma(x-\beta) \)</td><td>Learnable: ❌</td></tr>
    <tr><td>mReLU</td><td>\( f(x) = \min(\max(0, 1-x), \max(0, 1+x)) \)</td><td>Learnable: ❌</td></tr>
  </tbody>
</table>
