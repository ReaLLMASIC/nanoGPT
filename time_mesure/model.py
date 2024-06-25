#last modified: 12/26/2023
#   Descriptiom:
#   This is a revised version of nanoGpt model.py
#   Run this file to see the time took by each block
#   You can modify hyperparameters to see the influence of those paramaters to runtime
#   of each block
#
#   Note:
#   Please run !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#   to generate an input.txt as input texts
#
#
#
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
#list to store timings
time_qk = []
time_softmax = []
time_pv = []
time_proj = []
time_drop = []
time_ffw = []
time_ln1 = []
time_ln2 = []
# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions?
max_iters = 500
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        start_time_qk = time.time()
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        end_time_qk = time.time()
        time_taken_qk = end_time_qk - start_time_qk
        time_qk.append(time_taken_qk)
        print(f"Time taken for qk and qk transpose: {time_taken_qk} seconds\n")

        start_time_softmax = time.time()
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        end_time_softmax = time.time()
        time_taken_softmax = end_time_softmax - start_time_softmax
        time_softmax.append(time_taken_softmax)
        print(f"Time taken for softmax: {time_taken_softmax} seconds\n")

        start_time_pv = time.time()
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        end_time_pv = time.time()
        time_taken_pv = end_time_pv - start_time_pv
        time_pv.append(time_taken_pv)
        print(f"Time taken for P*V: {time_taken_pv} seconds\n")

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        start_time_proj = time.time()
        self.proj = nn.Linear(n_embd, n_embd)
        end_time_proj = time.time()
        time_taken_proj = end_time_proj - start_time_proj
        time_proj.append(time_taken_proj)
        #print(f"Time taken for proj: {time_taken_proj} seconds\n")

        start_time_drop = time.time()
        self.dropout = nn.Dropout(dropout)
        end_time_drop = time.time()
        time_taken_drop = end_time_drop - start_time_drop
        time_drop.append(time_taken_drop)
        #print(f"Time taken for drop: {time_taken_drop} seconds\n")

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)

        start_time_ffw = time.time()
        self.ffwd = FeedFoward(n_embd)
        end_time_ffw = time.time()
        time_taken_ffw = end_time_ffw - start_time_ffw
        time_ffw.append(time_taken_ffw)
        #print(f"Time taken for ffwd: {time_taken_ffw} seconds\n")

        start_time_ln1 = time.time()
        self.ln1 = nn.LayerNorm(n_embd)
        end_time_ln1 = time.time()
        time_taken_ln1 = end_time_ln1 - start_time_ln1
        time_ln1.append(time_taken_ln1)
        #print(f"Time taken for ln1: {time_taken_ln1} seconds\n")

        start_time_ln2 = time.time()
        self.ln2 = nn.LayerNorm(n_embd)
        end_time_ln2 = time.time()
        time_taken_ln2 = end_time_ln2 - start_time_ln2
        time_ln2.append(time_taken_ln2)
        #print(f"Time taken for ln2: {time_taken_ln2} seconds\n")

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    #sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

# Calculate and print the average timings after the training loop
avg_time_qk = sum(time_qk) / len(time_qk) if time_qk else 0
avg_time_softmax = sum(time_softmax) / len(time_softmax) if time_softmax else 0
avg_time_pv = sum(time_pv) / len(time_pv) if time_pv else 0
avg_time_proj = sum(time_proj) / len(time_proj) if time_proj else 0
avg_time_drop = sum(time_drop) / len(time_drop) if time_drop else 0
avg_time_ffw = sum(time_ffw) / len(time_ffw) if time_ffw else 0
avg_time_ln1 = sum(time_ln1) / len(time_ln1) if time_ln1 else 0
avg_time_ln2 = sum(time_ln2) / len(time_ln2) if time_ln2 else 0

print(f"Average Time for qk and qk transpose: {avg_time_qk} seconds")
print(f"Average Time for softmax: {avg_time_softmax} seconds")
print(f"Average Time for P*V: {avg_time_pv} seconds")
print(f"Average Time for projection: {avg_time_proj} seconds")
print(f"Average Time for dropout: {avg_time_drop} seconds")
print(f"Average Time for feedforward: {avg_time_ffw} seconds")
print(f"Average Time for linear1: {avg_time_ln1} seconds")
print(f"Average Time for linear2: {avg_time_ln2} seconds")
