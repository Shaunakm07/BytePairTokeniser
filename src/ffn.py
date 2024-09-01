import torch 
import torch.nn as nn
from src.gelu import GELU

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, weights=None):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)

        if weights != None:
            self.w1.weight.data = weights["w1.weight"]
            self.w2.weight.data = weights["w2.weight"]
        
    def forward(self, x):
        return (self.w2(GELU(self.w1(x))))