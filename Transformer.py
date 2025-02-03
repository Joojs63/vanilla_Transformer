import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
import torch
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512, #이런식으로 값을 할당하는 것은 hard coding이므로 객체를 생성할 때 이 값이 고정됨  
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.1,
        device= None,
        max_length=100,
    ):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            self.device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            self.device,
            max_length,
        )

    #src_mask는 src_pad_idx가 아닌 부분만 1로 채워진 텐서
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    #look ahead mask
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape #N과 trg_len이라는 변수에 값을 저장 a=trg.shape[0] 참고로 indexing은 [], 함수 값 대입은 ()로 함
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand( 
            N, 1, trg_len, trg_len
        )    #torch.ones는 1로 채워진 텐서를 생성하는 함수

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src) #src_mask는 src_pad_idx가 아닌 부분만 1로 채워진 텐서
        trg_mask = self.make_trg_mask(trg) #look ahead mask
        enc_src = self.encoder(src, src_mask) #인코더 block을 여러번 통과한 후 Decoder로 전달되는 값
        out = self.decoder(trg, enc_src, src_mask, trg_mask) #디코더 block을 여러번 통과한 후 최종 출력값
        return out


