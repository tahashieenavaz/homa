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
      <th>ID</th>
      <th>Activation</th>
      <th>Formula</th>
      <th>Learnable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>ADA</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ x e^x & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>2</td>
      <td>AOAF</td>
      <td>$$f(x) = \max(0,\, x - b \cdot a) + c \cdot a$$</td>
      <td>✅ (channel‑wise)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>AReLU</td>
      <td>$$f(x) = (1 + \sigma(b)) \cdot \max(0,\, x) + \mathrm{clamp}(a,\, 0.01,\, 0.99) \cdot \min(0,\, x)$$</td>
      <td>✅ (global)</td>
    </tr>
    <tr>
      <td>4</td>
      <td>ASiLU</td>
      <td>$$f(x) = \arctan\bigl(x \cdot \sigma(x)\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>5</td>
      <td>AbsLU</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ \alpha \, |x| & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>6</td>
      <td>BaseDLReLU</td>
      <td>$$f(z) = \begin{cases} z & z \ge 0 \\ a \cdot b_t \cdot z & z < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>7</td>
      <td>CaLU</td>
      <td>$$f(x) = x \left( \frac{\arctan(x)}{\pi} + 0.5 \right)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>8</td>
      <td>DLReLU</td>
      <td>Inherits BaseDLReLU</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>9</td>
      <td>DLU</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ \frac{x}{1 - x} & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>10</td>
      <td>DPReLU</td>
      <td>$$f(x) = \begin{cases} a\,x & x \ge 0 \\ b\,x & x < 0 \end{cases}$$</td>
      <td>✅ (channel‑wise)</td>
    </tr>
    <tr>
      <td>11</td>
      <td>DRLU</td>
      <td>$$f(x) = \max(0,\, x - \alpha)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>12</td>
      <td>DerivativeSiLU</td>
      <td>$$f(x) = \sigma(x)\bigl(1 + x(1 - \sigma(x))\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>13</td>
      <td>DiffELU</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ a\,(x e^x - b e^{b x}) & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>14</td>
      <td>DoubleSiLU</td>
      <td>$$f(x) = \frac{x}{1 + \exp\left( -\left(-\frac{x}{1 + e^{-x}}\right) \right)}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>15</td>
      <td>DualLine</td>
      <td>$$f(x) = \begin{cases} a\,x + m & x \ge 0 \\ b\,x + m & x < 0 \end{cases}$$</td>
      <td>✅ (channel‑wise)</td>
    </tr>
    <tr>
      <td>16</td>
      <td>EANAF</td>
      <td>$$f(x) = x \cdot g(h(x))$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Elliot</td>
      <td>$$f(x) = 0.5 + \frac{0.5\,x}{1 + |x|}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>18</td>
      <td>ExponentialDLReLU</td>
      <td>Inherits BaseDLReLU</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>19</td>
      <td>ExponentialSwish</td>
      <td>$$f(x) = e^{-x} \cdot \sigma(x)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>20</td>
      <td>FReLU</td>
      <td>$$f(x) = \begin{cases} x + b & x \ge 0 \\ b & x < 0 \end{cases}$$</td>
      <td>✅ (channel‑wise)</td>
    </tr>
    <tr>
      <td>21</td>
      <td>FlattedTSwish</td>
      <td>$$f(x) = \max(0,\, x)\cdot\sigma(x) + t$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>22</td>
      <td>GeneralizedSwish</td>
      <td>$$f(x) = x \cdot \sigma(e^{-x})$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>23</td>
      <td>Gish</td>
      <td>$$f(x) = x \cdot \ln\bigl(2 - e^{-e^x}\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>24</td>
      <td>IpLU</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ \frac{x}{1 + |x|^\alpha} & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>25</td>
      <td>LaLU</td>
      <td>$$f(x) = x \cdot \begin{cases}1 - 0.5 e^{-x} & x \ge 0 \\ 0.5 e^{x} & x < 0\end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>26</td>
      <td>LeLeLU</td>
      <td>$$f(x) = \begin{cases} a\,x & x \ge 0 \\ 0.01\,a\,x & x < 0 \end{cases}$$</td>
      <td>✅ (channel‑wise)</td>
    </tr>
    <tr>
      <td>27</td>
      <td>LogSigmoid</td>
      <td>$$f(x) = \ln\bigl(\sigma(x)\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>28</td>
      <td>Logish</td>
      <td>$$f(x) = x \cdot \ln\bigl(1 + \sigma(x)\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>29</td>
      <td>MSiLU</td>
      <td>$$f(x) = x\,\sigma(x) + \tfrac{1}{4} e^{-x^2 - 1}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>30</td>
      <td>MaxSig</td>
      <td>$$f(x) = \max\bigl(x, \sigma(x)\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>31</td>
      <td>MinSin</td>
      <td>$$f(x) = \min\bigl(x, \sin x\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>32</td>
      <td>NLReLU</td>
      <td>$$f(x) = \ln\bigl(1 + \beta \cdot \max(0, x)\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>33</td>
      <td>NReLU</td>
      <td>$$f(x) = \begin{cases} x + a & x \ge 0 \\ 0 & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>34</td>
      <td>NoisyReLU</td>
      <td>Inherits NReLU</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>35</td>
      <td>OAF</td>
      <td>$$f(x) = \max(0, x) + x \cdot \sigma(x)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>36</td>
      <td>PERU</td>
      <td>$$f(x) = \begin{cases} a\,x & x \ge 0 \\ a\,x\,e^{b x} & x < 0 \end{cases}$$</td>
      <td>✅ (channel‑wise)</td>
    </tr>
    <tr>
      <td>37</td>
      <td>PFLU</td>
      <td>$$f(x) = x \cdot 0.5\left(1 + \frac{x}{\sqrt{1 + x^2}}\right)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>38</td>
      <td>PLAF</td>
      <td>$$f(x) = \begin{cases} x - \delta & x \ge 1 \\ -x - \delta & x < -1 \\ \frac{|x|^d}{d} & -1 \le x < 1 \end{cases},\; \delta = 1 - \frac{1}{d}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>39</td>
      <td>Phish</td>
      <td>$$f(x) = x \cdot \tanh(\mathrm{GELU}(x))$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>40</td>
      <td>PiLU</td>
      <td>$$f(x) = \begin{cases} a\,x + c\,(1 - a) & x \ge c \\ b\,x + c\,(1 - b) & x < c \end{cases}$$</td>
      <td>✅ (channel‑wise)</td>
    </tr>
    <tr>
      <td>41</td>
      <td>PoLU</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ (1 - x)^{-\alpha} - 1 & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>42</td>
      <td>PolyLU</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ \frac{1}{1 - x} - 1 & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>43</td>
      <td>REU</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ x e^x & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>44</td>
      <td>RReLU</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ \frac{x}{a} & x < 0 \end{cases},\; a \in [\text{lower},\,\text{upper}]$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>45</td>
      <td>RandomizedSlopedReLU</td>
      <td>Inherits SlopedReLU</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>46</td>
      <td>ReCU</td>
      <td>Inherits RePU</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>47</td>
      <td>RePU</td>
      <td>$$f(x) = \max\bigl(0,\, x^\alpha\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>48</td>
      <td>ReQU</td>
      <td>Inherits RePU</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>49</td>
      <td>ReSP</td>
      <td>$$f(x) = \begin{cases} \alpha x + \ln 2 & x \ge 0 \\ \ln\bigl(1 + e^x\bigr) & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>50</td>
      <td>ReSech</td>
      <td>$$f(x) = x \cdot \mathrm{sech}(x)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>51</td>
      <td>SGELU</td>
      <td>$$f(x) = \alpha x \cdot \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>52</td>
      <td>SaRa</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ \frac{x}{1 + \alpha e^{-\beta x}} & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>53</td>
      <td>Serf</td>
      <td>$$f(x) = x \cdot \mathrm{erf}\bigl(\ln(1 + e^x)\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>54</td>
      <td>ShiLU</td>
      <td>$$f(x) = a \cdot \max(0,\, x) + b$$</td>
      <td>✅ (channel‑wise)</td>
    </tr>
    <tr>
      <td>55</td>
      <td>ShiftedReLU</td>
      <td>$$f(x) = \max(x, -1)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>56</td>
      <td>SiELU</td>
      <td>$$f(x) = x \cdot \sigma\bigl(2 \sqrt{\tfrac{2}{\pi}} (x + 0.044715\,x^3)\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>57</td>
      <td>SigLU</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ \frac{1 - e^{-2x}}{1 + e^{-2x}} & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>58</td>
      <td>SigmoidDerivative</td>
      <td>$$f(x) = e^{-x} \cdot \sigma(x)^2$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>59</td>
      <td>SinSig</td>
      <td>$$f(x) = x \cdot \sin\left(\frac{\pi}{2} \sigma(x)\right)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>60</td>
      <td>SineReLU</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ \epsilon (\sin x - \cos x) & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>61</td>
      <td>SlopedReLU</td>
      <td>$$f(x) = \begin{cases} \alpha x & x \ge 0 \\ 0 & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>62</td>
      <td>Smish</td>
      <td>$$f(x) = x \cdot \tanh\bigl(\ln(1 + \sigma(x))\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>63</td>
      <td>SoftModulusQ</td>
      <td>$$f(x) = \begin{cases} x^2(2 - |x|) & |x| \le 1 \\ |x| & |x| > 1 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>64</td>
      <td>SoftModulusT</td>
      <td>$$f(x) = x \cdot \tanh\left(\frac{x}{\alpha}\right)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>65</td>
      <td>SoftsignRReLU</td>
      <td>$$f(x) = \begin{cases} \frac{1}{(1 + x)^2} + x & x \ge 0 \\ \frac{1}{(1 + x)^2} + a x & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>66</td>
      <td>StarReLU</td>
      <td>$$f(x) = a (\max(0,\, x))^2 + b$$</td>
      <td>✅ (channel‑wise)</td>
    </tr>
    <tr>
      <td>67</td>
      <td>Suish</td>
      <td>$$f(x) = \max\bigl(x,\, x e^{-|x|}\bigr)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>68</td>
      <td>TBSReLU</td>
      <td>$$f(x) = x \cdot \tanh\left(\frac{1 - e^{-x}}{1 + e^{-x}}\right)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>69</td>
      <td>TSReLU</td>
      <td>$$f(x) = x \cdot \tanh(\sigma(x))$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>70</td>
      <td>TSiLU</td>
      <td>Let $\alpha = \frac{x}{1 + e^{-x}},\; f(x) = \frac{e^\alpha - e^{-\alpha}}{e^\alpha + e^{-\alpha}}$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>71</td>
      <td>TangentBipolarSigmoidReLU</td>
      <td>Inherits TBSReLU</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>72</td>
      <td>TangentSigmoidReLU</td>
      <td>Inherits TSReLU</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>73</td>
      <td>TanhExp</td>
      <td>$$f(x) = x \cdot \tanh(e^x)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>74</td>
      <td>TeLU</td>
      <td>$$f(x) = x \cdot \tanh(e^x)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>75</td>
      <td>ThLU</td>
      <td>$$f(x) = \begin{cases} x & x \ge 0 \\ \tanh\left(\frac{x}{2}\right) & x < 0 \end{cases}$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>76</td>
      <td>TripleStateSwish</td>
      <td>Define $a=\sigma(x), b=\sigma(x-\alpha), c=\sigma(x-\beta)$, then $$f(x) = x \cdot a (a + b + c)$$</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>77</td>
      <td>mReLU</td>
      <td>$$f(x) = \min\bigl(\max(0,\,1 - x),\,\max(0,\,1 + x)\bigr)$$</td>
      <td>❌</td>
    </tr>
  </tbody>
</table>
