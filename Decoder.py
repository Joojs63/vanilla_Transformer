import torch
import torch.nn as nn
from DecoderBlock import DecoderBlock

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        #하나의 단어를 512차원의 벡터로 embedding 함 
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        #위치벡터 추가
        self.position_embedding = nn.Embedding(max_length, embed_size)

        #마찬가지로 최종 선형변환 전에 디코더 block을 여러번 반복함.
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        #N: batch 크기(한번에 처리하는 문장개수수)
        N, seq_length = x.shape
        #arange(a,b) a부터 b-1까지의 수를 생성 expand는 텐서의 크기를 배치개수만큼 복사하여 크기를 늘리는 함수
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

