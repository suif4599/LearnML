import torch

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=3)

output = torch.tensor([[1, 2, 3, 4, 5],
                       [5, 2, 3, 4, 1],
                       [1, 3, 4, 2, 5]], dtype=torch.float)
target = torch.tensor([3, 0, 2])

print(loss_fn(output, target))