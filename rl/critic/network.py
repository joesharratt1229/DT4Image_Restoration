import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
from typing import Dict, Optional, Callable

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (
        depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (ResidualBlock, [2, 2, 2, 2]),
        '34': (ResidualBlock, [3, 4, 6, 3]),
        '50': (Bottleneck, [3, 4, 6, 3]),
        '101': (Bottleneck, [3, 4, 23, 3]),
        '152': (Bottleneck, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]


class TRelu(nn.Module):
    def __init__(self) -> None:
        super(TRelu, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
        self.alpha.data.fill_(0)
    
    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x
    

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                 in_planes: int,
                 out_planes: int, 
                 stride: int
                 ) -> None:
        
        super(ResidualBlock, self).__init__()


        self.conv1 = weightNorm(nn.Conv2d(in_planes, out_planes, kernel_size = 3, 
                                             stride = stride, padding = 1, bias = False))
        
        self.conv2 = weightNorm(nn.Conv2d(out_planes, out_planes, kernel_size = 3, 
                                          stride = stride, padding = 1, bias = False))

    
        

        self.relu_1 = TRelu()
        self.relu_2 = TRelu()
        
        self.shortcut = nn.Sequential()
        if (stride != 1) or (in_planes != (out_planes * self.expansion)):
            self.shortcut = weightNorm(nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                                      stride = stride, padding = 1, bias = False))

    def forward(self, x):
        out = self.relu_1(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu_2(out)
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

        self.conv1 = weightNorm(nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False))
            
        
        
        self.conv2 = weightNorm(nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False))
        
        
        self.conv3 = weightNorm(nn.Conv2d(out_planes, self.expansion * out_planes, 
                                          kernel_size=1, bias=False))

        
        
        self.relu_1 = TRelu()
        self.relu_2 = TRelu()
        self.relu_3 = TRelu()
        
        self.shortcut = nn.Sequential()
        if (stride) != 1 or (in_planes != self.expansion * out_planes):
            self.shortcut = weightNorm(
                nn.Conv2d(in_planes, 
                          self.expansion* out_planes, 
                          kernel_size=1,
                          stride = stride,
                          bias = False))
            
    
    def forward(self, x):
        output = self.relu_1(self.conv1(x))
        output = self.relu_2(self.conv2(output))
        output = self.conv3(output)
        output += self.shortcut(x)
        output = self.relu_3(output)
        return output

        


class Critic(nn.Module):
    #need to check this
    _num_inputs = 4

    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 depth: Dict,
                 ) -> None:
        
        super(Critic, self).__init__()
        block, num_blocks = cfg(depth)

        self.conv1 = weightNorm(nn.Conv2d(self._num_inputs, 
                                          self.in_planes,
                                          stride = 2,
                                          kernel_size = 3,
                                          padding = 1,
                                          bias = False))
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.fc = nn.Linear(512, num_outputs)
        self.relu_1 = TRelu()

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
    
    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        
        out = self.relu(self.conv1(x))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(x.size(0), -1))
        out = self.fc(out)
        return out

        
          
    


