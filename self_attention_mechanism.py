import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# 1) Input sentence
sentence='The quick brown fox jumps over a lazy dog'

# 2)splitting sentence into words with unique IDs
words={word:i for i,word in enumerate(sorted(sentence.split(" ")))}

# 3) Converting the integer indices into tensors which are later used to create embeddings
t_indices=torch.tensor([words[i] for i in words])

# 4) Converting the tensor indices into embeddings
vocab_size=50000 #assuming approximate large vocabulary size
embed=nn.Embedding(vocab_size,3) # 3-dimensonal embedding
embedded_sentence=embed(t_indices).detach() # each word embedded into a 3-D vector

# 5) Query,Key,Value matrices and calculaing query,key,value for all embedded words
embeddings_dimension=embedded_sentence.shape[1]
dim_query,dim_key,dim_value=2,2,4

W_query=torch.nn.Parameter(torch.rand(embeddings_dimension,dim_query))
W_key=torch.nn.Parameter(torch.rand(embeddings_dimension,dim_key))
W_value=torch.nn.Parameter(torch.rand(embeddings_dimension,dim_value))

query=embedded_sentence @ W_query 
key=embedded_sentence @ W_key 
value=embedded_sentence @ W_value

# 6) Calculating Attention Scores 
attention_scores=query @ key.T
attention_scores=attention_scores/math.sqrt(dim_key)
attention_weights=F.softmax(attention_scores,dim=-1) 


# 7) Generating the context vectors
context_vector= attention_weights @ value

# Including all of it in a Module
class SelfAttention(nn.Module):
    def __init__(self,embedding_dimension,dim_query,dim_key,dim_value):
        super(SelfAttention,self).__init__()
        self.embedding_dimension=embedding_dimension
        self.dim_query=dim_query
        self.dim_key=dim_key
        self.dim_value=dim_value
        self.W_query=torch.nn.Parameter(torch.rand(embedding_dimension,dim_query))
        self.W_key=torch.nn.Parameter(torch.rand(embedding_dimension,dim_key))
        self.W_value=torch.nn.Parameter(torch.rand(embedding_dimension,dim_value))

    def forward(self,x):
        Q=x @ self.W_query
        K=x @ self.W_key
        V=x @ self.W_value

        attention_scores=Q @ K.T/ math.sqrt(self.dim_key)
        attention_weights=F.softmax(attention_scores,dim=-1)
        context_vector=attention_weights @ V
        return context_vector

sa_module=SelfAttention(3,2,2,4)
output=sa_module(embedded_sentence)
print(output)








