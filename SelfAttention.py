import torch
import torch.nn as nn

#init은 내가 사용할 ouput을 도출하기 위한 기계, 또한 input을 정의해줌
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):          #selfattention의 초기화, 주로 선형변환
        super(SelfAttention, self).__init__()
        #재료 파트
        self.embed_size = embed_size  #128
        self.heads = heads            #4
        self.head_dim = embed_size // heads #32
        
        #만약 나눠지지 않으면 문장을 출력함
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        #기계 파트 
        #몇개의 뉴런이 input이고 몇개의 뉴런이 output인지 정해줌 거기에 들어가는 값은 아직 모름
        #input embedding을 받아서 선형변환을 통해 Q,K,V값을 도출
        self.values = nn.Linear(embed_size, embed_size)  #V
        self.keys = nn.Linear(embed_size, embed_size)    #Q
        self.queries = nn.Linear(embed_size, embed_size) #K
        self.fc_out = nn.Linear(embed_size, embed_size)  #attention 다 끝난 tensor를 한번 더 선형변환
        #각각에서의 Q K V값은 다 다름 가중치 값이 다르기 때문에
        
    #forward에서는 SelfAttention을 통해 원하는 값을 도출하는 과정을 정의해줌
    def forward(self, values, keys, query, mask): #얘네들은 이제 init 에서 정의된 self.attention(a,b,c)할때 호출됨
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
        query_key = torch.einsum("blhd,bqhd->bhlq", [queries, keys]) # 'b'atch 'l'ength 'h'eads 'd'imension 여기서 행렬곱을 하면 l*l 형태가 나옴
        # queries shape: (N, query_len, heads, heads_dim),          
        # keys shape: (N, key_len, heads, heads_dim)
        # energy(QK^T): (N, heads, query_len, key_len)

        # mask가 0인 부분은 -1e20으로 채워라 만약 mask를 None으로 설정하면 마스킹 안함
        if mask is not None:
            energy = query_key.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(query_key / (self.embed_size ** (1 / 2)), dim=3) #dim은 어느 방향으로 softmax를 적용할지 정함
        #여기서 dim = 3이라는 것은 마지막 차원을 기준으로 softmax를 적용한다는 것이고 이는 각 attention에 대해 row방향으로 1이되도록 확률을 정하는것임
        # attention shape: (N, heads, query_len, key_len) 2, 4, 9, 9 nhql

        # 결과가나오고 concatentation을 통해서 다시 원래의 shape로 만들어줌
        out = torch.einsum("bhlq,bqhd->blhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size) 2, 9, 128
        #각각의 독립적인 attention들을 FC layer를 통해 유기적으로 합쳐줌
        return out
    