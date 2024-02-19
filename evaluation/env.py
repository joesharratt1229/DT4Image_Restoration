import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from evaluation.utils.transformations import fft, ifft


class PnPEnv:
    def __init__(self, max_episode_step, denoiser) -> None:
        self.max_episode_step = max_episode_step
        self.denoiser = denoiser

    def reset(self, data, ddp, gpu_id):
        #data ()
        x = data['x0']
        x = torch.view_as_complex(x)
        z = x.clone().detach()
        u = torch.zeros_like(x)
        mask = data['mask'].reshape(1, 1, 128, 128).contiguous().to(torch.bool)
        y0 = data['y0'].contiguous()
        y0 = torch.view_as_complex(y0)
        gt = data['gt']
        if ddp:
            x, z, u, mask, y0, gt = x.to(gpu_id), z.to(gpu_id), u.to(gpu_id), mask.to(gpu_id), y0.to(gpu_id), gt.to(gpu_id)
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

        reward = self.compute_reward(x.real.squeeze(dim = 0), gt)
        states['x'] = x
        states['z'] = z
        states['u'] = u 

        return states, reward, done
    
    @staticmethod
    def get_policy_ob(state: OrderedDict):
        """
        x -> (1, 128, 128)
        """
        x = state['x'].real
        z = state['z'].real
        u = state['u'].real
        policy_ob = torch.cat((x, z, u), dim = 0)
        policy_ob = policy_ob.reshape(1, -1)
        return policy_ob


    @staticmethod
    def compute_reward(x, y0):
        x = x.reshape(128, 128)
        y0 = y0.reshape(128, 128)
        mse = torch.mean(F.mse_loss(x, y0, reduction = 'none'))
        psnr = 10 * torch.log10((1**2)/mse)
        return psnr.unsqueeze(dim = 0)


        
