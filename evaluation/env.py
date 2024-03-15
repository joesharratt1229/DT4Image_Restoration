import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from functools import partial

from collections import OrderedDict

from evaluation.utils.transformations import fft, ifft


class PnPEnv:
    def __init__(self, max_episode_step, denoiser, device_type) -> None:
        self.max_episode_step = max_episode_step
        self.denoiser = denoiser.to(device_type)

    def reset(self, data, device_type):
        #data ()
        x = data['x0']
        x = torch.view_as_complex(x)
        z = x.clone().detach()
        u = torch.zeros_like(x)
        mask = data['mask'].reshape(1, 1, 128, 128).contiguous().to(torch.bool)
        y0 = data['y0'].contiguous()
        y0 = torch.view_as_complex(y0)
        gt = data['gt']
 
        x, z, u , mask, y0, gt = x.to(device_type), z.to(device_type), u.to(device_type), mask.to(device_type), y0.to(device_type), gt.to(device_type)
        return OrderedDict({'x': x, 'y0': y0, 'z': z, 'u': u, 'mask': mask, 'gt': gt})

    
    def step(self, states: OrderedDict, action_dict: OrderedDict):
        T, mu, sigma_d = action_dict['T'], action_dict['mu'], action_dict['sigma_d']
        x, y0, z, u, mask, gt = states['x'], states['y0'], states['z'], states['u'], states['mask'], states['gt']

        temp_var = (z - u)
        x = self.denoiser(temp_var.real, sigma_d)
        z = fft(x + u)
        _mu = mu.view(1, 1, 1, 1)
        temp = ((_mu * z.clone()) + y0)/(1+ _mu)
        z[mask] = temp[mask]
        z = ifft(z)

        u = u + x - z

        if T > 0.5:
            done = True
        else:
            done = False

        states['x'] = x
        states['z'] = z
        states['u'] = u 

        return states, done
    
    @staticmethod
    def get_policy_ob(state: OrderedDict):
        """
        x -> (1, 128, 128)
        """ 
        policy_ob = state['x'].real
        policy_ob = policy_ob.reshape(1, -1)
        return policy_ob


    @staticmethod
    def compute_reward(x, y0):
        x = x.cpu().detach().numpy()
        y0 = y0.reshape(1, 128, 128).cpu().detach().numpy()
        x = x* 255
        y0 = y0 * 255
        return psnr_qrnn3d(x, y0)



class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = X[ch, :, :]
            y = Y[ch, :, :]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

def psnr_qrnn3d(X, Y, data_range=255):
    cal_bwpsnr = Bandwise(partial(peak_signal_noise_ratio, data_range=data_range))
    return np.mean(cal_bwpsnr(X, Y))     


        
