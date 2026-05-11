
import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

# Input token selection
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

# FUNCTIONS
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

def basic_self_attention_step_by_step():
    # Randomly intiated projection matrices
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
     
    # Compute query,  key, and value vectors
    query_2 = x_2 @ W_query
    key_2   = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2)

    # Compute key and value matrices for all tokens
    keys = inputs @ W_key
    values = inputs @W_value
    print("keys.shape: ", keys.shape)
    print("values.shape: ", values.shape)

    # Compute attention scores (query-key dot products)
    keys_2 = keys[1]
    attn_scores_2 = query_2 @ keys.T
    print(attn_scores_2)

    # Scaled attention scores and cmpute attention weights
    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
    print(f'Scaled attention weights:\n{attn_weights_2}')

    # Generate context Vector
    context_vec_2 = attn_weights_2 @ values
    print(context_vec_2)

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores=queries @ keys.T  # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

def main():
    torch.manual_seed(42)
    print('\nBasic:')
    basic_self_attention_step_by_step()

    # Self Attention
    torch.manual_seed(42)
    print('\nSelf Attention:')
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))







if __name__ == "__main__":    
   main()
