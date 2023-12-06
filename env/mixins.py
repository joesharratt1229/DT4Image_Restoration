from typing import Dict, Tuple
import numpy as np
import torch
import json
#custom
from utils.transformations import fft, ifft, complex2channel


class CSMRIMixin:
    def _build_init_csmri_ob(self,
                              batch: torch.Tensor,
                              noise_level: int,
                              mask : torch.Tensor
                              ) -> Tuple[torch.Tensor, Dict, Dict]:
        
        B, C, H, W = batch.shape
        
        y0 = fft(batch)
        y0, noise_level = self.noise_mod(y0, noise_level)
        mask = mask.repeat(B, 1, 1, 1)
        y0 *= mask
        Aty0 = ifft(y0)

        variables = self.solver.reset(Aty0)
        noise_map = torch.ones_like(mask) * noise_level
        T = torch.zeros_like(noise_map)
        env_ob = {
            'variables': variables,
            'y0': y0,
            'mask': mask,
            'Aty0': Aty0,
            'T': T,
            'noise_map': noise_map
        }
        return (torch.cat([variables.real, complex2channel(y0), Aty0.real, mask, T, noise_map], dim = 1),
                env_ob)
                
    
    def _build_next_csmri_ob(self,
                             env_ob: Dict
                             ) -> Tuple[torch.Tensor, Dict]:
        
        variables, y0, Aty0, mask, T, noise_map = env_ob['variables'], env_ob['y0'], env_ob['Aty0'], env_ob['mask'], env_ob['T'], env_ob['noise_map']
        return (torch.cat([variables.real, complex2channel(y0), Aty0.real, mask, T, noise_map], dim = 1),
                env_ob)




        











        