import torch
import torch.nn as nn
from SelfAttention import SelfAttention
from TransformerBlock import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        #nn.Module에 있는 init을 상속받겠다.
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads=heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, src_mask, trg_mask):
        #masked multi head self attention
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        #Encoder부분에서와 같이 attention-add norm-FFN-add norm을 거침
        out = self.transformer_block(value, key, query, src_mask)
        return out