# Overview of Vision Transformers

The goal here is to understand ViT

- ref https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py

## Embeddings

Init
- get number of patches and grid size from config file
- Get dropuout from config
- position embedding size = #patches * hidden size
- cls_token size = hidden size

Forward Pass
- Get patch embedding but passing through a Conv2D using the kernel size, stride = patch size
- Add classification token
- Add dropout

## Encoder

Init
- Get number of layer from config file
- Loop through the number of layers and build "blocks"

Forward Pass
- Get hidden states and weights
- Perform Layer normalization 
- Return normalized hidden states and attention weights

## Block

Init
- Get hidden size, 
- Layer normalization for feed forward and attention layer

Forward Pass for a block
- Apply layer normalization
- Apply attention
- Apply skip connection
- Apply feed forward
- Apply skip connection

## MLP

Feed Forward Network

Init
- Build 2 FC layers 
- "gelu" activation function
- Get dropout from config

Forward Pass 
- FC layers
- Gelu activation
- Apply dropout
- FC 
- Apply dropout


## Attention

Init
- Get attention heads
- Initialize Linear network for query, key, value
- Get droput values from config

Forward Pass 
- Apply Linear network for query
- Apply Linear network for key
- Apply Linear network for value
- Calcaute attention scores by performing matmul of key and query 
- Get probabilties using softmax
- Apply dropout 
- Get values using matmul of probabilities and values
- Create new context layer
- Apply another Linear network get attention output
- return attention and weights



