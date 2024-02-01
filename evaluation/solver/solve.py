import torch  
import torch.nn as nn

from denoiser.base import UNetDenoiser2D
from utils.transformations import fft, ifft

class BaseSolver(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.denoiser = UNetDenoiser2D()
    
class AdmmSolver(BaseSolver):
    def __init__(self) -> None:
        #check if this is correct
        super().__init__()
    
    def reset(self, data):
        x = data.clone()
        z = x.clone()         
        u = torch.zeros_like(x)         
        return torch.cat((x, z, u), dim=1)
    
    def forward(self, observations, actions):
        variables, y0, mask = observations
        x, z, u = torch.chunk(variables, chunks = 3, dim = 1)

        mask = mask.long()
        sigma_d  = actions['sigma_d']
        mu = actions['mu']

        assert mu.shape == torch.size([x.shape[0], 1, 1, 1]), 'Error mu incorrect dimensions'

        temp = z - u 
        x = self.denoiser(temp.real, sigma_d)

        z = fft(x + u)
        #_mu = mu[:, 0].view(x.shape[0], 1, 1, 1)
        temp = ((mu * z.clone()) + y0)/ (1 + mu)
        z[mask] = temp[mask]
        z = ifft(z)

        u = u + x - z

        return torch.cat((x, z, u), dim = 1)





