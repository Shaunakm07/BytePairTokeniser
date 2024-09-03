import torch 
import torch.nn as nn 
from src.scaled_dot_product_attention import scaled_dot_product_attention

class multihead_self_attention(nn.Module):
    def __init__(self, d_model, number_heads, attn_pdrop):
        super().__init__()
        self.q_weights = nn.Linear(d_model, d_model, bias=False)
        self.k_weights = nn.Linear(d_model, d_model, bias=False)
        self.v_weights = nn.Linear(d_model, d_model, bias=False)

        self.number_heads = number_heads
        self.attn_pdrop = attn_pdrop
        self.head_size = int(d_model/number_heads)

        self.o_layer = nn.Linear(d_model, d_model)
    def forward(self, x):
        Q, K, V = self.q_weights(x), self.k_weights(x), self.v_weights(x)
        print(Q.shape)
        Q = list(torch.split(Q, self.head_size, 1))
        Q = torch.stack(Q, 1)

        K = list(torch.split(K, self.head_size, 1))
        K = torch.stack(K, 1)

        V = list(torch.split(V, self.head_size, 1))
        V = torch.stack(V, 1)

        attention = scaled_dot_product_attention(K, Q, V)
        return self.o_layer(attention)