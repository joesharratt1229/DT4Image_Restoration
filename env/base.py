import torch
import torch.nn as nn
import numpy as np
import os
from scipy.io import loadmat
from typing import Any, Dict
from typing import Tuple

from denoiser.base import UNetDenoiser2D
from env.mixins import CSMRIMixin
from utils.masks import generate_radial_mask, cartesian_mask, variable_density_mask

class PnPEnv(CSMRIMixin):
    mask_dictionary = {
        'radial': generate_radial_mask,
        'cartesian': cartesian_mask,
        'variable_density': variable_density_mask
    }
    def __init__(self, 
                 noise_model: nn.Module,
                 solver: Any
                 ) -> None:
        
        self.noise_mod = noise_model
        self.solver = solver
        self._max_episode_step = 6

    def build_init_ob(self, 
                        batch: torch.Tensor,
                        task_dict: Dict
                        )-> Tuple[torch.Tensor, Dict]:
        
        #mask_func = task_dict['mask']
        mask_str = task_dict['mask']
        mask_func = self.mask_dictionary[mask_str]
        noise_level = task_dict['noise_level']
        if mask_func == 'variable_density':
            min_density = task_dict['min_density']
            max_density = task_dict['mask_density']
            mask = mask_func(min_density = min_density, max_density = max_density)
        else:
            acceleration = task_dict['acceleration']
            mask = mask_func(acceleration)
        
        mask = torch.from_numpy(mask).bool()
        policy_ob, env_ob = self._build_init_csmri_ob(batch, noise_level, mask)
        return policy_ob, env_ob
        
    
    def build_next_ob(self,
                     observation: Dict
                     ) -> Tuple[torch.Tensor, Dict]:
        return self._build_next_csmri_ob(observation)

