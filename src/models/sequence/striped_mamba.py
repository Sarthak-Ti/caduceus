

#a python module that contains all necessary components for my striped mamba. Including rotary embeddings, MHA with gating, Hydra core, and the full stacked model.

from models.nn.MHA import MHAGate, Qwen3RMSNorm
#this is the core MHA block with gating


#trasnformer block