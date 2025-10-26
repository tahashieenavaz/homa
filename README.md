## Device Management

```py
from homa import cpu, mps, cuda, device

torch.tensor([1, 2, 3, 4, 5]).to(cpu())
torch.tensor([1, 2, 3, 4, 5]).to(cuda())
torch.tensor([1, 2, 3, 4, 5]).to(mps())
torch.tensor([1, 2, 3, 4, 5]).to(device())
```
