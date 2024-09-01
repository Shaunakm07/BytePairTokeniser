import torch 
import torch.nn as nn

class rms_norm(nn.Module):
    def __init__(self, d_model, eps, weight):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = weight["weight"]
    
    def forward(self, x):
        squared_eps = torch.square(x)
        mean_squared = torch.mean(squared_eps, dim=-1, keepdim=True) + self.eps
        sqrt = torch.sqrt(mean_squared)
        return (x/sqrt) * self.weight