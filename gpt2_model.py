"""
Minimal GPT-2 architecture used as a benchmark model.
"""


# Benchmark Model

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import math
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

# GPT-2 architecture

# Define some constants
VOCAB_SIZE = 50257 # The size of the GPT-2 vocabulary
EMBED_DIM = 768 # The dimension of the token and positional embeddings
NUM_LAYERS = 12 # The number of transformer layers
NUM_HEADS = 12 # The number of attention heads
FFN_DIM = 3072 # The dimension of the feed-forward network
MAX_LEN = 1024 # The maximum length of the input sequence
DROP_RATE = 0.1 # The dropout rate

# Define the GPT-2 model class
class GPT2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ffn_dim, max_len, drop_rate):
        super(GPT2, self).__init__()
        # Initialize the token and positional embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_len, embed_dim)
        # Initialize the transformer layers
        self.transformer_layers = nn.ModuleList([TransformerLayer(embed_dim, num_heads, ffn_dim, drop_rate) for _ in range(num_layers)])
        # Initialize the output layer
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)
        # Tie the weights of the output layer and the token embeddings
        self.output_layer.weight = self.token_embeddings.weight
        # Initialize the dropout layer
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, input_ids, attention_mask):
        # Get the batch size and the sequence length
        batch_size, seq_len = input_ids.shape
        # Get the token embeddings
        token_embeds = self.token_embeddings(input_ids)
        # Get the positional embeddings
        position_ids = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(input_ids.device)
        position_embeds = self.position_embeddings(position_ids)
        # Add the token and positional embeddings
        embeddings = token_embeds + position_embeds
        # Apply dropout
        embeddings = self.dropout(embeddings)
        # Initialize the hidden state
        hidden_state = embeddings
        # Iterate over the transformer layers
        for layer in self.transformer_layers:
            # Apply the layer and get the output
            hidden_state = layer(hidden_state, attention_mask)
        # Apply the output layer and get the logits
        logits = self.output_layer(hidden_state)
        return logits

# Define the transformer layer class
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, drop_rate):
        super(TransformerLayer, self).__init__()
        # Initialize the multi-head self-attention layer
        self.attention = MultiHeadAttention(embed_dim, num_heads, drop_rate)
        # Initialize the feed-forward network
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, drop_rate)
        # Initialize the layer normalization layers
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        # Initialize the dropout layer
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, hidden_state, attention_mask):
        # Apply the attention layer and get the output and the attention weights
        attention_output, attention_weights = self.attention(hidden_state, hidden_state, hidden_state, attention_mask)
        # Add and normalize
        hidden_state = self.ln1(hidden_state + self.dropout(attention_output))
        # Apply the feed-forward network and get the output
        ffn_output = self.ffn(hidden_state)
        # Add and normalize
        hidden_state = self.ln2(hidden_state + self.dropout(ffn_output))
        return hidden_state

# Define the multi-head attention layer class
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_rate):
        super(MultiHeadAttention, self).__init__()
        # Check if the embed dimension is divisible by the number of heads
        assert embed_dim % num_heads == 0, "The embed dimension must be divisible by the number of heads"
        # Initialize the head dimension
        self.head_dim = embed_dim // num_heads
        # Initialize the number of heads
        self.num_heads = num_heads
        # Initialize the query, key and value projection layers
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        # Initialize the output projection layer
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        # Initialize the dropout layer
        self.dropout = nn.Dropout(drop_rate)
        # Initialize the scaling factor
        self.scale = np.sqrt(self.head_dim)

    def forward(self, query, key, value, attention_mask):
        # Get the batch size
        batch_size = query.shape[0]
        # Project the query, key and value
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        # Reshape the query, key and value to [batch_size, num_heads, seq_len, head_dim]
        query = query.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Compute the attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        # Apply the attention mask
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        # Compute the attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        # Compute the attention output
        attention_output = torch.matmul(attention_weights, value)
        # Reshape the attention output to [batch_size, seq_len, embed_dim]
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        # Project the attention output
        attention_output = self.output_proj(attention_output)
        return attention_output, attention_weights

# Define the feed-forward network class
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, drop_rate):
        super(FeedForwardNetwork, self).__init__()
        # Initialize the linear layers
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        # Initialize the dropout layer
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, hidden_state):
        # Apply the first linear layer and the gelu activation
        hidden_state = F.gelu(self.linear1(hidden_state))
        # Apply dropout
        hidden_state = self.dropout(hidden_state)
        # Apply the second linear layer
        hidden_state = self.linear2(hidden_state)
        return hidden_state

# Define a function to load the GPT-2 125M model checkpoints
def load_model(model, model_path):
    # Load the state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # Remove the 'transformer.' prefix from the keys
    state_dict = {k.replace('transformer.', ''): v for k, v in state_dict.items()}
    # Load the state dict to the model
    model.load_state_dict(state_dict)
    return model

# Define a function to generate text using the model
def generate_text(model, tokenizer, input_text, max_len, temperature, top_k, top_p):
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # Get the input length
    input_len = input_ids.shape[-1]
    # Check if the input length is less than the maximum length
    assert input_len < max_len, "The input length must be less than the maximum length"
    # Generate text
    output_ids = model.generate(input_ids, max_length=max_len, temperature=temperature, top_k=top_k, top_p=top_p)
    # Decode the output ids
    output_text = tokenizer.decode(output_ids[0])
    return output_text

# Define the model path
model_path = 'gpt2-pytorch_model.bin'
# Define the tokenizer
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2')
# Instantiate the model
model = GPT2(VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, FFN_DIM, MAX_LEN, DROP_RATE)
# Load the model checkpoints
model = load_model(model, model_path)
# Define the input text
input_text = "Hello, world!"
# Define the generation parameters
max_len = 50
temperature = 0.9
top_k = 50
top_p = 0.9
# Generate text
output_text = generate_text(model, tokenizer, input_text, max_len, temperature, top_k, top_p)
# Print the output text
print(output_text)
