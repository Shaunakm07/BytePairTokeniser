import torch
import math

def GELU(x):   
    return x * 1/2 * (1 + torch.erf(x / math.sqrt(2)))