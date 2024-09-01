import torch

def softmax_operation(input, dim):
    max = input.max(dim=dim, keepdim=True).values
    tensor = input - max.expand(input.shape)
    return torch.exp(tensor) / torch.sum(torch.exp(tensor), dim, keepdim=True)