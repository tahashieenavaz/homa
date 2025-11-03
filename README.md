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
