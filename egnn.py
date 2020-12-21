import torch
import torch.nn as nn
from layers import Encoder, Decoder

class EGNN(nn.Module):
  def __init__(self,
               src_attr_size,
               src_max_len,
               tgt_attr_size,
               tgt_max_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2):
    super(Transformer, self).__init__()

    self.encoder = Encoder(src_attr_size, src_max_len, num_layers, model_dim,
                            num_heads, ffn_dim, dropout)
    self.decoder = Decoder(tgt_attr_size, tgt_max_len, num_layers, model_dim,
                            num_heads, ffn_dim, dropout)

    self.linear = nn.Linear(model_dim, tgt_attr_size, bias=False)
    self.softmax = nn.Softmax(dim=2)

