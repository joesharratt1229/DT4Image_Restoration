import torch
import torch.nn.functional as F

import torchvision.transforms as transforms

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from collections import OrderedDict

from evaluation.utils.transformations import fft, ifft
from functools import partial

def complex2channel(x):
    N, C, H, W, _ = x.shape
    # N C H W 2 -> N 2C H W
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    x = x.view(N, C*2, H, W)
    return x
    
def greyscale_to_rgb(tensor):
    # Repeat the single channel three times along the channel dimension
    dim = tensor.shape[1]
    zeros_tensor = torch.zeros(2, dim, dim)
    rgb_tensor = torch.cat((tensor, zeros_tensor), dim = 0)
    return rgb_tensor



class PnPEnv:
    def __init__(self, max_episode_step, denoiser, device_type) -> None:
        self.max_episode_step = max_episode_step
        self.denoiser = denoiser.to(device_type)
        self._load_no_ref()
        
    def _load_no_ref(self):
        model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                       regressor_dataset="kadid10k")
        model.eval()
        self.no_ref_model = model
        
    def run_no_ref_reward(self, state):
        img = state['x']
        img = img.reshape(1,128, 128)
        
        
        
        img_ds = transforms.Resize((img.size(1) // 2, img.size(2)// 2))(img)
        img_ds = greyscale_to_rgb(img_ds)
        img = greyscale_to_rgb(img)
       
        with torch.no_grad(), torch.cuda.amp.autocast():
           score = self.no_ref_model(img.unsqueeze(dim = 0), img_ds.unsqueeze(dim = 0), return_embedding=False, scale_score=True)
        return score.mean(0).item() 
        
        
    def reset(self, data, device_type):
        #data ()
        x = data['x0']
        x = torch.view_as_complex(x)
        data['complex_y0'] = data['y0']
        z = x.clone().detach()
        u = torch.zeros_like(x)
        mask = data['mask'].reshape(1, 1, 128, 128).contiguous().to(torch.bool)
        y0 = data['y0'].contiguous()
        y0 = torch.view_as_complex(y0)
        gt = data['gt']
        Aty0 = data['ATy0'][..., 0]
 
        x, z, u , mask, y0, gt = x.to(device_type), z.to(device_type), u.to(device_type), mask.to(device_type), y0.to(device_type), gt.to(device_type)
        return OrderedDict({'x': x, 'y0': y0, 'z': z, 'u': u, 'mask': mask, 'gt': gt, 'ATy0': Aty0, 'T': 0, 'complex_y0': data['y0']})

    
    def step(self, states: OrderedDict, action_dict: OrderedDict):
        T, mu, sigma_d = action_dict['T'], action_dict['mu'], action_dict['sigma_d']
 
        x, y0, z, u, mask, gt = states['x'], states['y0'], states['z'], states['u'], states['mask'], states['gt']
        
        if T > 0.5:
            done = True
            return states, done
        else:
            done = False

        temp_var = (z - u)
        x = self.denoiser(temp_var.real, sigma_d)
        z = fft(x + u)
        _mu = mu.view(1, 1, 1, 1)
        temp = ((_mu * z.clone()) + y0)/(1+ _mu)
        z[mask] = temp[mask]
        z = ifft(z)

        u = u + x - z

        states['x'] = x
        states['z'] = z
        states['u'] = u 
        states['T'] = states['T'] + 1/30

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
        x = x.cpu().detach()
        y0 = y0.reshape(1, 128, 128).cpu().detach()
        return torch_psnr(x, y0)



def torch_psnr(output, gt):
    N = output.shape[0]
    output = torch.clamp(output.real, 0, 1)
    mse = torch.mean(F.mse_loss(output.view(N, -1), gt.view(N, -1), reduction='none'), dim=1)
    psnr = 10 * torch.log10((1 ** 2) / mse)
    return psnr.unsqueeze(1)

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


        
