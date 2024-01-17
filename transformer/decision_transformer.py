import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class CausalAttention(nn.Module):
    def __init__(self, config) -> None:
        
        super().__init__()
        assert config.embed_dim % config.n_heads == 0, 'Incompatible embedding dimensions for number of attention heads'
        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.o_proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.att_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)


        self.register_buffer('masking', torch.tril(torch.ones(config.block_size, 
                                                              config.block_size)).
                                                              view(1, 1, 
                                                                   config.block_size, 
                                                                   config.block_size))

        #self._reset_parameters()

    def _reset_parameters(self):
        pass

    def forward(self, x):
        B, T, E = x.size()
        q, k, v = self.qkv_proj(x).split(self.embed_dim, dim = 2)

        k = k.view(B, T, self.n_heads, E//self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, E//self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, E//self.n_heads).transpose(1, 2)

        attn = (q @ k.transpose(-1, -2))/math.sqrt(q.size(-1))
        attn = attn.masked_fill(self.masking[..., :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim = -1)
        attn = self.att_dropout(attn)

        y = attn @ v #b, att_n, input_seq, head_dim
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        y = self.resid_dropout(self.o_proj(y))
        print(y.shape)
        return y 
    

    
class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.embed_dim, config.embed_dim * 4)
        self.fc_proj = nn.Linear(config.embed_dim * 4, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.fc_proj(x)
        x = self.dropout(x)
        return x



class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.c_att = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.c_att(self.ln1(x)) #communicate phase
        x = self.mlp(self.ln2(x)) #compute phase
        return x


    
class DecisionTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.state_dim = config.state_dim
        self.actor_dim = config.actor_dim

        self.embed_dim = config.embed_dim
        n_heads = config.n_heads

        self.time_embed = nn.Embedding(config.max_episode_length, config.embed_dim)
        self.task_embed = nn.Embedding(config.num_tasks, config.embed_dim)
        #TODO should activatation function be added after embedd action and returns?
        self.embed_action = nn.Linear(self.actor_dim, self.embed_dim)
        self.embed_return = nn.Linear(1, self.embed_dim)

        self.state_encoder = nn.Sequential(
            nn.Conv2d(3, 23, 8, stride = 4, padding = 0), nn.ReLU(),
            nn.Conv2d(32, 64, 6, stride = 2, padding = 0), nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(), nn.Linear(6400, config.embed_dim), nn.Tanh())
        
        blocks = [Block(config) for _ in range(config.n_blocks)]

        self.transformer = nn.Sequential(*blocks)

    def _init_weights(self):
        pass

    def forward(self, actions, returns, T, states):
        pass


        

        


        



        






class Config:
    def __init__(self, embed_dim, n_heads, block_size):
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.n_heads = n_heads
        self.dropout =  0.01

# Create a configuration instance
config = Config(embed_dim=512, n_heads=8, block_size=10)



attn_obj = CausalAttention(config=config)










