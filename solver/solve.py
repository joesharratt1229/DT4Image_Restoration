import torch  

from denoiser.base import UNetDenoiser2D
from utils.transformations import fft, ifft
    
class AdmmSolver:
    iter_num = 5
    def __init__(self) -> None:
        #check if this is correct
        self.denoiser = UNetDenoiser2D()
    
    def reset(self, data):
        x = data.clone()
        z = x.clone()         
        u = torch.zeros_like(x)         
        return torch.cat((x, z, u), dim=1)
    
    def __call__(self, observations, actions):
        variables, y0, mask = observations
        x, z, u = torch.chunk(variables, chunks = 3, dim = 1)

        mask = mask.long()
        sigma_d  = actions['sigma_d']
        mu = actions['mu']

        for i in range(self.iter_num):
            temp = z - u 
            x = self.denoiser(temp.real, sigma_d[:, i])

            z = fft(x + u)
            _mu = mu[:, i].view(x.shape[0], 1, 1, 1)
            temp = ((_mu * z.clone()) + y0)/ (1 + _mu)
            z[mask] = temp[mask]
            z = ifft(z)

            u = u + x - z

        return torch.cat((x, z, u), dim = 1)





