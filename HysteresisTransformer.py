import torch
import torch.nn as nn
import math

def positional_encoding(seq_len, embed_size):
    """
    Sinusoidal positional encoding을 계산합니다.
    입력:
      - seq_len: 시퀀스 최대 길이
      - embed_size: 임베딩 차원
    출력:
      - (seq_len, embed_size) 크기의 고정 positional encoding
    """
    pe = torch.zeros(seq_len, embed_size)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2, dtype=torch.float) * (-math.log(10000.0) / embed_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Self-Attention 모듈 (multi-head attention)
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embed size must be divisible by heads"
        
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys   = nn.Linear(embed_size, embed_size, bias=False)
        self.queries= nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        values  = self.values(values)
        keys    = self.keys(keys)
        queries = self.queries(queries)
        
        # (N, len, embed_size) -> (N, len, heads, head_dim)
        values  = values.reshape(N, value_len, self.heads, self.head_dim)
        keys    = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        
        
        # energy: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
            
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        
        # out: (N, query_len, heads, head_dim) -> (N, query_len, embed_size)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

# Transformer 블록 (Self-attention + Feed Forward)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1     = nn.LayerNorm(embed_size)
        self.norm2     = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Encoder: 변위(displacement) 데이터를 입력받아 선형투영 + 고정 positional encoding 적용
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):

        """ 입력:
          - input_dim: 변위 데이터의 feature 차원 (예: 1)
          - embed_size: 임베딩 차원
          - num_layers: Transformer 블록 수
          - heads: multi-head attention head 수
          - device: 디바이스
          - forward_expansion: feed-forward 확장 비율
          - dropout: dropout 비율
          - max_length: 시퀀스 최대 길이 """

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device     = device
        self.input_projection = nn.Linear(input_dim, embed_size) #input의 마지막 차원이 embed_size로 선형변환됨 
        # 고정된 sin/cos positional encoding (학습되지 않음)
        pe = positional_encoding(max_length, embed_size)  # (max_length, embed_size)
        self.register_buffer("positional_encoding1", pe.unsqueeze(0))     
        """ (1, max_length, embed_size) 이미 
            텐서에 값들이 maxlength까지 저장되어있음 """
        
        self.layers = nn.ModuleList(
            [TransformerBlock(
                embed_size, heads, dropout, forward_expansion) 
             for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # x: (N, seq_length, input_dim)
        N, seq_length  = x.shape[0], x.shape[1]
        x = self.input_projection(x)                      # (N, seq_length, embed_size)
        x = x + self.positional_encoding1[:, :seq_length, :]  
        """ positional encoding이라는 이름은 Buffer이름과 같아야함
        Broadcast를 통해 같은 positional encoding값을 x의 배치마다다 더함"""
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x

# Decoder Block: self-attention(힘 데이터) + cross-attention(변위 정보) 적용
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        # Decoder self-attention (힘 데이터 처리)
        self.self_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        # Cross-attention: Encoder의 출력과 결합
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, cross_mask, self_mask):
        # self-attention: 힘 데이터에 대해 패딩 및 lookahead mask 적용
        self_attn = self.self_attention(x, x, x, self_mask)
        query = self.dropout(self.norm(self_attn + x))
        # Cross-attention: 현재 시점 이전의 변위 정보만 참조하도록 mask 적용
        out = self.transformer_block(value, key, query, cross_mask)
        return out

# Decoder: 힘(force) 데이터를 예측하기 위한 모듈  
# 입력은 <start> 토큰이 포함된 힘 시퀀스이며, 최종 출력은 연속값 (예: 1차원 force)
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        """
        입력:
          - output_dim: 힘 데이터의 feature 차원 (예: 1)
          - embed_size: 임베딩 차원
          - num_layers: Transformer 블록 수
          - heads: multi-head attention head 수
          - forward_expansion: feed-forward 확장 비율
          - dropout: dropout 비율
          - device: 디바이스
          - max_length: 시퀀스 최대 길이
        """
        super(Decoder, self).__init__()
        self.device = device
        self.output_projection = nn.Linear(output_dim, embed_size)
        pe = positional_encoding(max_length, embed_size)
        self.register_buffer("positional_encoding2", pe.unsqueeze(0))
        
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)]
        )
        # 최종 출력: 임베딩 차원 -> 출력 차원 (force)
        self.fc_out = nn.Linear(embed_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, cross_mask, self_mask):
        # x: (N, seq_length, output_dim)
        N, seq_length = x.shape[0], x.shape[1]
        x = self.output_projection(x)                      # (N, seq_length, embed_size)
        x = x + self.positional_encoding2[:, :seq_length, :] # positional encoding 추가
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, cross_mask, self_mask)
        out = self.fc_out(x)  # (N, seq_length, output_dim)
        return out

# Transformer: Encoder와 Decoder를 결합하여 변위(Encoder) → 힘(Decoder) 예측 수행
class Transformer(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        src_pad_value, 
        trg_pad_value,
        embed_size=128, 
        num_layers=6, 
        forward_expansion=4, 
        heads=4,
         dropout=0, 
         device="cuda", 
         max_length=3000
    ):
        """
        입력:
          - input_dim: Encoder 입력 차원 (예: 변위는 1차원)
          - output_dim: Decoder 출력 차원 (예: 힘은 1차원)
          - src_pad_value: Encoder의 패딩 값 (예: 0.0)
          - trg_pad_value: Decoder의 패딩 값 (예: 0.0)
          - embed_size, num_layers, forward_expansion, heads, dropout, max_length: 하이퍼파라미터
        """
        self.src_pad_value = src_pad_value
        self.trg_pad_value = trg_pad_value
        self.device = device

        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            input_dim, 
            embed_size, 
            num_layers, 
            heads, 
            device, 
            forward_expansion, 
            dropout,
            max_length
        )
        
        self.decoder = Decoder(
            output_dim, 
            embed_size,
            num_layers, 
            heads, 
            forward_expansion, 
            dropout, 
            device, 
            max_length
        )
        
    def make_src_mask(self, src):
        # src: (N, L, input_dim) — 패딩 값은 모든 feature가 src_pad_value (예: 0.0)
        pad_mask = (src.abs().sum(dim=-1) != self.src_pad_value).unsqueeze(1).unsqueeze(2)  # (N,1,1,L)
        L = src.shape[1]
        lookahead_mask = torch.tril(torch.ones(L, L, device=src.device)).unsqueeze(0).unsqueeze(0)
        return pad_mask * lookahead_mask

    def make_trg_mask(self, trg):
        # trg: (N, L, output_dim)
        pad_mask = (trg.abs().sum(dim=-1) != self.trg_pad_value).unsqueeze(1).unsqueeze(2)
        L = trg.shape[1]
        lookahead_mask = torch.tril(torch.ones(L, L, device=trg.device)).unsqueeze(0).unsqueeze(0)
        return pad_mask * lookahead_mask

    def make_cross_mask(self, trg, src):
        # trg: (N, trg_len, output_dim), src: (N, src_len, input_dim)
        N, trg_len, _ = trg.shape
        N, src_len, _ = src.shape
        device = trg.device
        allowed = (torch.arange(src_len, device=device).unsqueeze(0) <= 
                   torch.arange(trg_len, device=device).unsqueeze(1)).float()  # (trg_len, src_len)
        allowed = allowed.unsqueeze(0).unsqueeze(0)  # (1,1,trg_len,src_len)
        encoder_pad_mask = (src.abs().sum(dim=-1) != self.src_pad_value).unsqueeze(1).unsqueeze(2).float()  # (N,1,1,src_len)
        return encoder_pad_mask * allowed

    def forward(self, src, trg):
        """
        src: 변위 시퀀스, shape (N, seq_length, input_dim)
        trg: 힘 시퀀스, shape (N, seq_length, output_dim)
             단, Decoder 입력은 <start> 토큰이 포함되어 있으며, teacher forcing 시 trg[:, :-1, :]가 사용됨
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        cross_mask = self.make_cross_mask(trg, src)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, cross_mask, trg_mask)
        return out

# main: 배치=30, 시퀀스 길이=3000, 임베딩 차원=128, head=4
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 30
    seq_length = 3000
    input_dim = 1   # 변위는 1차원 연속값
    output_dim = 1  # 힘도 1차원 연속값

    # 패딩 값은 0.0 (만약 실제 데이터에서 0.0이 valid value라면 다른 값을 선택)
    src_pad_value = 0.0
    trg_pad_value = 0.0

    model = Transformer(
        input_dim, output_dim,
        src_pad_value, trg_pad_value,
        embed_size=128, num_layers=6, forward_expansion=4,
        heads=4, dropout=0, device=device, max_length=seq_length
    ).to(device)

    # 더미 입력 데이터 생성
    # Encoder: 변위 시퀀스, shape (batch, seq_length, 1)
    src = torch.randn(batch, seq_length, input_dim).to(device)
    
    # Decoder: 힘 시퀀스, 첫 timestep은 <start> 값 (여기서는 1.0으로 고정), 나머지는 임의의 값
    trg = torch.randn(batch, seq_length, output_dim)
    trg[:, 0, :] = 1.0  # <start> 토큰 값
    trg = trg.to(device)
    
    # teacher forcing: Decoder 입력은 trg[:, :-1, :]
    output = model(src, trg[:, :-1, :])
    print(output.shape)  
    # 예상 출력 shape: (batch, seq_length-1, output_dim)
