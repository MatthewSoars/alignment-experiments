import torch

vocab_size = 256
d_model = 64
max_seq_len = 128

def tokenization(sentence): # Tokenization - turning text into numbers
    return list(map(int, sentence.encode("utf-8")))


x = torch.tensor([tokenization("test")], dtype=torch.long)  # Dummy Tensor fine it doesn't click yet

tok_emb = torch.nn.Embedding(vocab_size, d_model)  # Creating the token embedding table
x_tok = tok_emb(x)  # (B, T, D)
print(x_tok.shape)  # (1, 2, 64)

pos_emb = torch.nn.Embedding(max_seq_len, d_model)  # Creating the positional embeddings table
B, T = x.shape
pos_ids = torch.arange(T, dtype=torch.long)          # (T,)
pos_vecs = pos_emb(pos_ids)                          # (T, D)


x_rep = x_tok + pos_vecs.unsqueeze(0)  # Adding Tokens and positional vectors together (B, T, D)
print(x_rep.shape)

print(x_rep[0, 0, :5]) # Seeing if it works
