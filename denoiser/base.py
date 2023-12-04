import os
import torch
import torch.nn as nn
from typing import Tuple

from .model.unet import UNet

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class UNetDenoiser2D(nn.Module):
    def __init__(self) -> None:
        super(UNetDenoiser2D, self).__init__()
        print(CURRENT_DIR)
        self.check_path = os.path.join(CURRENT_DIR, 'weights/unet-nm.pt')
        net = UNet(2, 1)
        net.load_state_dict(torch.load(self.check_path, map_location = torch.device('cpu')))
        net.eval()

        for param in net.parameters():
            param.requires_grad = False
        
        self.net = net

    
    def forward(self,
                x: torch.Tensor,
                sigma: int
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        N, C, H, W = x.shape

        sigma = sigma.view(-1, 1, 1, 1)

        noise_map = torch.ones(N, 1, H, W).to(x.device) * sigma
        out = self.net(torch.cat([x, noise_map], dim = 1))

        return torch.clamp(out, 0, 1)