from src.softmax import softmax_operation
import torch 
import math 

def scaled_dot_product_attention(K, Q, V, mask = None, pdrop=0.0):
    q_k = Q @ K.transpose(-1, -2)
    logits = q_k / math.sqrt(V.shape[-1])
    mask = mask.float()
    mask[mask==1] = -float("inf")
    mask[mask==0] = 1
    logits = mask * logits
    softmax = softmax_operation(logits, dim=-1)

    return softmax @ V