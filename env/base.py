import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from scipy.io import loadmat
from typing import Any, Dict
from typing import Tuple

from denoiser.base import UNetDenoiser2D
from env.mixins import CSMRIMixin
from utils.masks import generate_radial_mask, cartesian_mask, variable_density_mask
from utils.transformations import complex2channel

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
        self.max_episode_step = 6
        self.episode_num = 0
        self.done = 1

    def build_init_ob(self, 
                    batch: torch.Tensor,
                    task_dict: Dict
                    )-> Tuple[torch.Tensor, Dict]:
        
        B = batch.shape[0]
        self.idx_done = torch.arange(0, B)
        mask_str = task_dict['mask']
        mask_func = self.mask_dictionary[mask_str]
        noise_level = task_dict['noise_level']
        if mask_str == 'variable_density':
            min_density = task_dict['min_density']
            max_density = task_dict['max_density']
            mask = mask_func(min_density = min_density, max_density = max_density)
        else:
            acceleration = task_dict['acceleration']
            mask = mask_func(acceleration)
        
        mask = torch.from_numpy(mask).bool()
        policy_ob, env_ob = self._build_init_csmri_ob(batch, noise_level, mask)

        self.episode_num = 0 
        return policy_ob, env_ob
    

    def reset(self, 
              observations: Dict,
              device 
              ) -> Tuple[torch.Tensor, Dict]:
        B = observations['variables'].shape[0]
        self.idx_done = torch.zeros(B)
        for key, value in observations.items():
            observations[key] = value.to(device)
        variables, y0, Aty0, mask, T, noise_map = observations['variables'], observations['y0'], observations['Aty0'], observations['mask'], observations['T'], observations['noise_map']
        T = torch.zeros_like(noise_map)
        observations['T'] = T
        return (torch.cat([variables.real, complex2channel(y0), Aty0.real, mask, T, noise_map], dim = 1),
                observations)

        
    def step(self,
             observation: Dict,
             done: torch.Tensor
             ) -> Tuple[torch.Tensor, Dict]:
        observation['T'] += 1/self.max_episode_step
        self.idx_done[done == 1] = 1
        self.episode_num += 1
        return self._build_next_csmri_ob(observation), self.idx_done
    
    def compute_reward(self, observation_tuple):
        observation, gt = observation_tuple
        metric = self._compute_metric(observation, gt)
        reward = metric - self.last_metric
        self.last_metric = metric
        return reward
    
    def _compute_metric(self, observation: torch.Tensor, gt: torch.Tensor):
        x, _, _ = torch.chunk(observation, chunks = 3, dim = 1)
        assert x.shape == gt.shape
        def prepare(image = torch.Tensor) -> torch.Tensor:
            N = image.shape[0]
            image = torch.clamp(image, 0, 1)
            image = image.view(N, -1)
            return image

        x = prepare(x.real)
        gt = prepare(gt.real)
        mse = F.mse_loss(x, gt, reduction = 'none').mean(dim = 1)
        psnr = 10 * torch.log10((1 ** 2) / mse)
        return psnr
    

    def build_state_action_ob(self, variables: torch.Tensor, env_ob: Dict) -> torch.Tensor:
        _, y0, Aty0, mask, T, noise_map = env_ob['variables'], env_ob['y0'], env_ob['Aty0'], env_ob['mask'], env_ob['T'], env_ob['noise_map']
        return torch.cat([variables.real,  complex2channel(y0), Aty0.real, mask, T, noise_map], dim = 1)
    


