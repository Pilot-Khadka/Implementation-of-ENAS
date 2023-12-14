import torch
import torch.nn as nn
import math

class MultiheadedAttention(nn.Module):
    def __init__(self,input_dim, model_dim,batch_size,sequence_length, num_heads):
        super(MultiheadedAttention, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.num_heads = num_heads

        # all k,q,v vector concatenated
        self.kqv = nn.Linear(input_dim, 3*model_dim)
        self.linear = nn.Linear(model_dim,model_dim)

    def forward(self,x):
        qkv = self.kqv(x)
        print(qkv.size())
        head_dim = model_dim // num_heads
        qkv = qkv.reshape(self.batch_size,self.sequence_length,self.num_heads, 3* head_dim)
        # [batch_size, num_head, sequence_length, 3*head_dim]
        qkv = qkv.permute(0,2,1,3)
        # breakdown by last dim
        q,k,v = qkv.chunk(3, dim=-1)

        # size of vector
        d_k = q.size()[-1]
        scaled = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(d_k)
        scaled.shape
        attention = torch.softmax(scaled, dim=-1)
        #value vector for every single attention head
        values = torch.matmul(attention,v)

        # concatenate and make it 512 dimension
        values = values.reshape(self.batch_size,self.sequence_length,self.num_heads*head_dim)

        out = self.linear(values)

        return out

if __name__=="__main__":
    sequence_length = 4
    batch_size = 1
    input_dim = 512
    model_dim = 512 # output of model for every single word
    num_heads = 8
    x = torch.randn(batch_size, sequence_length, input_dim)
    print(x.size())
    attn = MultiheadedAttention(input_dim,model_dim,batch_size,sequence_length,num_heads)
    qkv = attn(x)
    # batchsize, sequence length,output_dimension
    print(qkv.size())
