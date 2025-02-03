import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,#몇개의 단어가 올 수 있는지 모든 가지수
        embed_size, #하나의 단어를 512차원의 벡터로 embedding 함
        num_layers, #몇개의 transformer block을 쌓을 것인지
        heads, #몇개의 head를 사용할 것인지
        device,
        forward_expansion, #.expand와는 다른 것. FFN에서 사용되는 hidden layer의 크기를 결정하는 변수
        dropout,
        max_length, #positional embedding을 위한 변수
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size #굳이 있어야할까? forward에서 사용되지도 않는데
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size) 
        #nn.Embedding은 (src_vocab_size, embed_size) 크기의 가중치 행렬을 생성하며, 각 단어 인덱스에 해당하는 벡터를 가져올 수 있음.
    #여기서는 위치 임베딩을 학습가능한 요소로 보았음. 내가 하는거에는 그냥 고정 위치 인코딩 해주면됨
        self.position_embedding = nn.Embedding(max_length, embed_size)
    #nn.ModuleList**는 PyTorch 모듈(nn.Module)로 이루어진 리스트를 정의
    #각 반복에서 **TransformerBlock**을 생성하여 리스트에 추가
    #self.layers = nn.ModuleList()
    #for _ in range(num_layers):
    #self.layers.append(
    #TransformerBlock(embed_size, heads, dropout, forward_expansion)
    # ) 랑 동일한 역할을 함함
        self.layers = nn.ModuleList(
            [TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion)
                for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        #expand는 텐서의 크기를 배치개수만큼 복사하여 크기를 늘리는 함수
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )
        #입력 x(정수 인덱스)를 받아, 해당하는 임베딩 벡터를 반환
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
    #여기서의 out은 인코더 block을 여러번 통과한 후 Decoder로 전달되는 값