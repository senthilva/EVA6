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

## Block

## MLP

## Attention

## Encoder

Init
- Get number of layer from config file
- Loop through the number of layers and build "blocks"

Forward Pass
- Get hidden states and weights
- Perform Layer normalization 
- Return normalized hidden states and attention weights



