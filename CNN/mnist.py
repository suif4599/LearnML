import os
import torch
from torchvision.datasets.mnist import MNIST

def get_loader(batch_size=32):
    mnist = MNIST(root=os.path.join(os.path.dirname(__file__), "raw"), download=False, train=True)
    dataset = torch.utils.data.TensorDataset(mnist.data.unsqueeze(1).float() / 256, mnist.targets)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mnist = MNIST(root=os.path.join(os.path.dirname(__file__), "raw"), download=False, train=False)
    dataset = torch.utils.data.TensorDataset(mnist.data.unsqueeze(1).float() / 256, mnist.targets)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

# ld, _ = get_loader()
# for x, y in ld:
#     print(x)
#     print(y)
#     break