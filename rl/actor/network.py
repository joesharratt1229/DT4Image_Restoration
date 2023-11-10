import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import OrderedDict
from copy import deepcopy
from typing import Optional, Callable, Dict, Tuple


def config(depth: list
           ) -> Tuple:
    depth_lst = [18, 34, 50, 101, 152]
    assert depth in depth_lst, 'Error: depth type is incorrect'
    cf_dict = {
        '18': (ResidualBlock, [2,2,2,2]),
        '34': (ResidualBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                 in_planes: int,
                 out_planes: int, 
                 stride: int,
                 bnorm: Optional[Callable[..., nn.Module]] = None
                 ) -> None:
        
        super(ResidualBlock, self).__init__()
        if bnorm is None:
            bnorm = nn.SyncBatchNorm

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_planes, out_planes, kernel_size = 3, 
                        stride = stride, padding = 1, bias = False),
                        bnorm(out_planes),
                        nn.ReLU()
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_planes, out_planes, kernel_size = 3, 
                        stride = stride, padding = 1, bias = False),
                        bnorm(out_planes))
        self.shortcut = nn.Sequential()
        if (stride != 1) or (in_planes != (out_planes * self.expansion)):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride = stride, padding = 1, bias = False),
                bnorm(out_planes * self.expansion)
            )
            

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, 
                 in_planes: int, 
                 out_planes: int, 
                 stride: int,
                 bnorm: Optional[Callable[..., nn.Module]]
                 ) -> None:
        
        super(Bottleneck, self).__init__()
        if bnorm is None:
            bnorm = nn.SyncBatchNorm

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            bnorm(out_planes),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False),
            bnorm(out_planes),
            nn.ReLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes, self.expansion * out_planes, kernel_size=1, bias=False),
            bnorm(out_planes))
        
        self.shortcut = nn.Sequential()
        if (stride) != 1 or (in_planes != self.expansion * out_planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, 
                          self.expansion* out_planes, 
                          kernel_size=1,
                          stride = stride,
                          bias = False))
            
    
    def forward(self, x):
        output = self.conv2(self.conv1(x))
        output += self.shortcut(x)
        output = F.relu(output)
        return output
        



class Policy(nn.Module):
    _num_inputs = 7
    #Three determinisitc actions
    _num_actions = 3
    _action_ranges_dictionary = {
        'mu': {'scale': 1, 'shift': 0},
        'sigma_d': {'scale': 70 / 255, 'shift': 0},
        'tau': {'scale': 2, 'shift': 0}
    }

    def __init__(self,
                 action_bundle: int,
                 depth: Dict) -> None:
        
        super(Policy, self).__init__()
        block, layers = config(depth)
        self.action_bundle = action_bundle

        self.in_planes = 64
        self.conv1 = nn.Sequential(nn.Conv2d(self._num_inputs, 
                                             64, 
                                             stride = 2, 
                                             kernel_size = 3,
                                             padding = 1, 
                                             bias = False),
                                   nn.SyncBatchNorm(64),
                                   nn.ReLU())
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride = 2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

        self.deterministic = nn.Sequential(
            nn.Linear(512, self.num_actions * action_bundle),
            nn.Sigmoid()
        )

        self.probabilistic = nn.Sequential(
            nn.Linear(512, 2),
            nn.Softmax(dim = 1)
        )

        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode = 'fan_out')
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight_, 1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, 
                    block: nn.Sequential, 
                    planes: int, 
                    num_blocks: int, 
                    stride: int) -> nn.Sequential:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  

        return nn.Sequential(*layers) 
    
    @property
    def adaptation_parameters(self
                              ) -> torch.Parameter:
        return self.parameters()
    
    
    def mapping(self, actions: torch.Tensor) -> OrderedDict:


        action_split = actions.shape[1] // self._num_actions
        actions = torch.split(actions, 
                    split_size_or_sections = action_split,
                    dim = -1)
        
        output_dic = OrderedDict()
        
        for i, key in enumerate(self._action_ranges_dictionary):
            output_dic[key] = actions[i] * (self._action_ranges_dictionary[key]['scale']/
                                            self._action_ranges_dictionary[key]['shift'])
            
        return output_dic
        
    
    def forward(self,
                x: torch.Tensor,
                idx_stop: Optional[int],
                train: bool,
                params: Optional[nn.Module] = None) -> Tuple:
        
        if params is not None:
            self.soft_update(params)
        
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        time_probs = self.probabilistic(x)
        deterministic_actions = self.deterministic(x)

        time_dist = Categorical(time_probs)
        entropy = time_dist.entropy().unsqueeze()

        if idx_stop is None:
            if train:
                idx_stop = time_dist.sample()
            else:
                idx_stop = torch.argmax(time_probs, dim = 1)

        action_logprob = time_dist.log_prob(idx_stop).unsqueeze(1)
        action = self.mapping(deterministic_actions)

        return action, action_logprob, entropy
    
    
    def update_params_inner(self):
        pass











