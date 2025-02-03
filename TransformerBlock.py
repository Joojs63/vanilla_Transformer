import torch.nn as nn
from SelfAttention import SelfAttention
# forward_expansion은 FFN에서 사용되는 hidden layer의 크기를 결정하는 변수
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        #init에서는 Selfattention의 init을 형식에 따라 호출
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
#nn.LayerNorm**은 레이어별로 학습 가능한 가중치와 편향을 가짐 따라서 각각 정의해줘야됨
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        #일부 입력 뉴런(벡터값)을 0으로 만들어버림 - 과적합 방지
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out