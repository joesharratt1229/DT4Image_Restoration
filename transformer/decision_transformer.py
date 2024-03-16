"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from dataclasses import dataclass
from collections import OrderedDict


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
        self.action_dim = config.action_dim

        self.embed_dim = config.embed_dim
        
        self.time_embed = nn.Embedding(config.max_timestep, config.embed_dim)
        self.embed_dropout = nn.Dropout(config.embd_dropout)
        
        self.task_embed = nn.Embedding(9, config.embed_dim)

        #self.embed_time = nn.Embedding(config.max_episode_length, config.embed_dim)
        #TODO should activatation function be added after embedd action and returns?
        self.embed_action = nn.Sequential(nn.Linear(self.action_dim, self.embed_dim),
                                          nn.Tanh())
        self.embed_return = nn.Sequential(nn.Linear(1, self.embed_dim),
                                          nn.Tanh())
        #self.embed_task = nn.Embedding(config.num_tasks, config.embed_dim)
        self.layer_n = nn.LayerNorm(config.embed_dim)

        self.state_encoder = nn.Sequential(
                                nn.Conv2d(1, 8, 8, stride = 4, padding = 0), nn.ReLU(),
                                nn.Conv2d(8, 16, 4, stride = 2, padding = 0), nn.ReLU(),
                                nn.Conv2d(16, 16, 3, stride = 1, padding = 0), nn.ReLU(),
                                nn.Flatten(), nn.Linear(2304, config.embed_dim), nn.Tanh())
        
        blocks = [Block(config) for _ in range(config.n_blocks)]

        self.transformer = nn.Sequential(*blocks)

        self.predict_action = nn.Sequential(
            nn.Linear(self.embed_dim, self.action_dim),
            nn.Sigmoid()
        )
        
        self.predict_rtg = nn.Linear(self.embed_dim, 1)

        self.apply(self._init_weights)

        self.action_range = OrderedDict({'T': {'scale': 1, 'shift': 0} ,
                                         'sigma_d': {'scale': 70 / 255, 'shift': 0},
                                         'mu': {'scale': 1, 'shift': 0}})

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean = 0.0, std = 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.beta)
        return optimizer
    

    def forward(self, rtg, states, timesteps, task, actions = None, eval_rtg = False, eval_actions = False): 
        #actions (batch, block_size, 3)
        #rtgs(batch, block_size, 1)
        #T (batch, block_size, 1)
        #states (batch, block_size, (1 * 128 * 128)
        
        batch_size, block_size, _ = states.size()
        rtg_embeddings = self.embed_return(rtg) 
        state_embeddings = self.state_encoder(states.reshape(-1, 1, 128, 128).contiguous())
        state_embeddings = state_embeddings.reshape(batch_size, block_size, -1)
        timesteps = timesteps.to(torch.int64).reshape(batch_size, -1)

        timesteps_embeddings = self.time_embed(timesteps)
        #TODO whether to concatenate task embeddings or add as we have done
        task_embeddings = self.task_embed(task)
        state_embeddings = state_embeddings + task_embeddings
        
        
        if actions is not None:
            action_embeddings = self.embed_action(actions)
            token_embeddings = torch.zeros((batch_size, 3 * block_size, self.embed_dim), device = state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
            timesteps_interleaved = torch.repeat_interleave(timesteps_embeddings, 3, dim = 1)

        else:
            token_embeddings = torch.zeros((batch_size, 2 * block_size, self.embed_dim), device = state_embeddings.device)
            token_embeddings[:, ::2, :] = rtg_embeddings
            token_embeddings[:, 1::2, :] = state_embeddings
            timesteps_interleaved = torch.repeat_interleave(timesteps_embeddings, 2, dim = 1)
            
        x = self.embed_dropout(token_embeddings + timesteps_interleaved)
        x = self.transformer(x)

        #TODO -> does it need a layer norm
        x = self.layer_n(x)
        
        #TODO reshape 
        #compute this loss on its ass
        if actions is not None:
            pred_actions = self.predict_action(x[:, 1::3, :])
            pred_rtg = self.predict_rtg(x[:, 2::3, :])
        else:
            pred_actions = self.predict_action(x[:, 1::2, :])
            
            
        pred_actions, action_dict = self._transform_actions(pred_actions, timesteps)
        
        
        if eval_rtg:
            return pred_rtg
        elif (eval_actions is True) or (actions is None):
            return pred_actions, action_dict
        else:
            return pred_actions, pred_rtg, action_dict
        
    
    def _transform_actions(self, outputs, time):
        chunk_size = outputs.shape[-1]//self.action_dim
        action_values = torch.split(outputs, chunk_size, dim = -1)
        action_dict = OrderedDict()
        for i, key in enumerate(self.action_range):
            action_dict[key] = action_values[i] * self.action_range[key]['scale']\
                        + self.action_range[key]['shift']
                         
        outputs = torch.cat([action_dict[key] for key in action_dict.keys()], dim = -1)
        return outputs, action_dict

        

class DecisionTransformerConfig:
    dropout = 0.1
    embd_dropout = 0.1
    #batch_size = 32
    embed_dim = 128
    n_heads = 4
    action_dim = 3
    max_timestep = 30
    n_blocks = 5

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)




