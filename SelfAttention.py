import torch
import torch.nn as nn

#init은 내가 사용할 ouput을 도출하기 위한 기계, 또한 input을 정의해줌
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):          #selfattention의 초기화
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  #128
        self.heads = heads            #4
        self.head_dim = embed_size // heads #32
        
        #만약 나눠지지 않으면 문장을 출력함
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        #몇개의 뉴런이 input이고 몇개의 뉴런이 output인지 정해줌 거기에 들어가는 값은 아직 모름
        self.values = nn.Linear(embed_size, embed_size) #단일 선형 변환 기계를 만들어줌 wx+b 
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        #각각에서의 Q K V값은 다 다름 가중치 값이 다르기 때문에에
    #forward에서는 SelfAttention을 통해 원하는 값을 도출하는 과정을 정의해줌
    def forward(self, values, keys, query, mask): #얘네들은 이제 self.mmm(a,b,c)할때 호출됨됨
        # Get number of training examples
        N = query.shape[0] #batch size

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        #아직 차원 변환하기 전 Q K V 값을 생성 각 값들은 input embedding(out out out)값들임
        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # reshape을 통해서 embed_size(512)를 heads(8)와 head_dim(64)로 나누어줌
        values = values.reshape(N, value_len, self.heads, self.head_dim)   # 2, 9, 4, 32
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)         # 2, 9, 4, 32
        queries = queries.reshape(N, query_len, self.heads, self.head_dim) # 2, 9, 4, 32 nlhd

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        
        #query_key = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) 
        query_key = torch.einsum("blhd,bqhd->bhlq", [queries, keys]) # 'b'atch 'l'ength 'h'eads 'd'imension 여기서 행렬곱을 하면 l*l 형태가 나옴옴
        # queries shape: (N, query_len, heads, heads_dim),          
        # keys shape: (N, key_len, heads, heads_dim)
        # energy(QK^T): (N, heads, query_len, key_len)

        # mask가 0인 부분은 -1e20으로 채워라 만약 mask를 None으로 설정하면 마스킹 안함
        if mask is not None:
            energy = query_key.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(query_key / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len) 2, 4, 9, 9 nhql

        # out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
        #     N, query_len, self.heads * self.head_dim
        # )
        # 결과가나오고 concatentation을 통해서 다시 원래의 shape로 만들어줌
        out = torch.einsum("bhlq,bqhd->blhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size) 2, 9, 128

        return out