import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

conv = nn.Conv2d

class ConvLayer(nn.Sequential):
    def __init__(self,
                 conv: nn.Module,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: Optional[int] = None,
                 dilation: int = 1,
                 norm: Optional[nn.Module] = None,
                 act: Optional[nn.Module] = None,
                 bias: bool = True
                 ) -> None:
        
        super(ConvLayer, self).__init__()

        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
        if act is not None:
            self.add_module('act', act)

class ConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 k: Optional[int] = 3,
                 s: Optional[int] = 1,
                 act: Optional[nn.Module] = nn.LeakyReLU(0.2),
                 num_layers: Optional[int] = 3) -> None:
          
          super(ConvBlock, self).__init__()
          self.add_module('conv-0', ConvLayer(
              conv, in_channels, channels, k, s, padding= None,
              dialation = 1, norm = None, act = act))
          
          for i in range(1, num_layers):
              self.add_module('conv-{}'.format(i), ConvLayer(
                  conv, channels, channels, k, s,
                  padding = None, dilation = 1, norm = None, act= act))
              

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 requires_grad: Optional[bool] = False) -> None:
        
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(512 + 256, 256)
        self.up2 = up(256 + 128, 128)
        self.up3 = up(128 + 64, 64)
        self.up4 = up(64 + 32, 32)
        self.outc = outconv(32, out_channels)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    
    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        
        noisy_img = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        residual = self.outc(x)

        C = residual.shape[1]
        return noisy_img[:, :C, ...] + residual
    

class inconv(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int) -> None:
        super(inconv, self).__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, 
                x) -> torch.Tensor:
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int) -> None:
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch)
        )

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x = self.mpconv(x)
        return x
    
class up(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 bilinear: Optional[bool] = True) -> None:
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode= 'bilinear', align_corners = True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride = 2)

        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor
                ) -> torch.Tensor:
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x2.size()[3]

        x1 = F.pad(x1, (diffX //2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2))
        
        x = torch.cat([x2, x1], dim = 1)
        x = self.conv(x)
        return x
    
class outconv(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int) -> None:
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x
    


    

    


        



        


        

              
            




        
