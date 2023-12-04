from typing import Any
import torch
import numpy as np

class ContinuousNoise:
    def __init__(self, 
                 minimum_noise: int = 5, 
                 maximum_noise: int = 20
                 ) -> None:
        
        self.minimum_noise = minimum_noise
        self.maxiumum_noise = maximum_noise

    
    def __call__(self, x, noise):
        assert (noise > self.minimum_noise) & (noise < self.maxiumum_noise), 'Value Error: Invalid noise level'

        sigma = noise/255
        y = x + torch.randn(*x.shape) * sigma

        return y, sigma

        