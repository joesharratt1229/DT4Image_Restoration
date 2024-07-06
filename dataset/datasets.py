import os
from abc import abstractmethod
import torch
import torch.utils.data.dataset as dataset
import json
import numpy as np
from scipy.io import loadmat
import h5py
import re



concat_pad = lambda x, padding_len: torch.cat([x, torch.zeros(([padding_len] + list(x.shape[1:])), dtype = x.dtype)], dim = 0)

class BaseDataset(dataset.Dataset):
    _tasks = ['5', '10', '15']
    _task_tokenizer = {task: i for i, task in enumerate(_tasks)}
    
    _min_rtg = -1.08
    _max_rtg = 16.6
    
    
    def __init__(self, block_size, data_dir, action_dim) -> None:
        super(BaseDataset, self).__init__()
        self.block_size = block_size
        self.data_dir = data_dir
        self.action_dim = action_dim

    def __len__(self):
        return len(os.listdir(self.data_dir))
    
    def _normalize_rtg(self, rtg_arr):
        minmax_norm = lambda x: (x - BaseDataset._min_rtg)/(BaseDataset._max_rtg - BaseDataset._min_rtg)
        new_rtg = [minmax_norm(x) for x in rtg_arr]
        return new_rtg
    
    @abstractmethod
    def __getitem__(self, index):
        pass


def extract_task(s):
    pattern = r'\d+_\d+'
    match = re.search(pattern, s)
    return match.group()
    

class TrainingDataset(BaseDataset):

    def __init__(self, block_size, data_dir, action_dim, state_file_path) -> None:
        super().__init__(block_size, data_dir, action_dim)
        self.state_file_path = state_file_path
        self.timestep_max = 30   
    
    def _get_image(self,trajectory):
        traj_path = trajectory[10:]
        with h5py.File(self.state_file_path, 'r') as file:
            data = np.float32(file[traj_path][:]/255)
        image = torch.from_numpy(data)
        return image


    def _get_states(self, state_path_list, pad = None):
        state_tensors = []

        for trajectory in state_path_list:
            x = self._get_image(trajectory)
            state_tensors.append(x)

        states = torch.stack(state_tensors)

        if pad is not None:
            states = torch.cat([states, torch.zeros(([pad] + list(states.shape[1:])), dtype = x.dtype)])

        seq_len = states.shape[0]
        states = states.view(seq_len, -1)

        return states
    
    @staticmethod
    def _get_actions(action_dict, traj_start, traj_end, pad = None):
        action_lis = [torch.Tensor(action_dict[key][traj_start:traj_end]) for key in action_dict.keys()]
              
        actions = torch.stack((action_lis), dim = 1)
        
        if pad is not None:
            padding = torch.zeros((pad, actions.shape[1]))
            actions = torch.cat([actions, padding], dim = 0)

        return actions

    def __getitem__(self, 
                    index: int
                    ) -> torch.Tensor:
        block_size = self.block_size
        traj_name = os.listdir(self.data_dir)[index]
        traj_path = os.path.join(self.data_dir, traj_name)
        
        with open(traj_path, 'r') as file:
            traj_dict = json.load(file)
            
        traj_len = len(traj_dict['RTG'])
        
        #acceleration = traj_dict['acceleration']
        #noise_level = traj_dict['noise_level']
        
        #task = acceleration[0] + '_' + noise_level
        #task = noise_level
        
        #task = self._task_tokenizer[task]
        #task = torch.tensor([task])
        #task = task.repeat(block_size)
        traj_dict['RTG'] = self._normalize_rtg(traj_dict['RTG'])
        

        if traj_len >= block_size:
            if traj_len==block_size:
                start = 0
            else:
                start = np.random.randint(0, traj_len - block_size)

            actions = self._get_actions(traj_dict['Actions'], start, start + block_size)
            rtg = np.array(traj_dict['RTG'][start:start+block_size])
            rtg = torch.from_numpy(rtg).type(torch.float32).reshape(-1, 1)
            timesteps = torch.arange(start, start + block_size).reshape(-1, 1)
            state_path_list = traj_dict['State Paths'][start: start + block_size]
            states = self._get_states(state_path_list)
            traj_masks = torch.ones(block_size)
        else:
            padding_len = block_size - traj_len
            actions = self._get_actions(traj_dict['Actions'], 0, traj_len, pad = padding_len)  
            rtg = np.array(traj_dict['RTG'])
            rtg = torch.from_numpy(rtg).type(torch.float32).reshape(-1, 1)
            rtg = torch.cat([rtg, torch.zeros(([padding_len] + list(rtg.shape[1:])), dtype = rtg.dtype)], dim = 0)
            traj_masks = torch.cat([torch.ones(traj_len),torch.zeros(padding_len)], dim = 0)
            state_path_list = traj_dict['State Paths'][:traj_len]
            states = self._get_states(state_path_list, pad = padding_len)
            timesteps = torch.arange(start = 0, end = block_size).reshape(-1, 1)
        
        traj_masks = traj_masks.unsqueeze(dim = -1)
        return states, actions, rtg, traj_masks, timesteps#, task


class EvaluationDataset(BaseDataset):

    def __init__(self, block_size, data_dir, action_dim, rtg_target) -> None:
        super().__init__(block_size, data_dir, action_dim)
        self.rtg_target = rtg_target
        self.fns = [im for im in os.listdir(self.data_dir) if im.endswith('.mat')]
        self.fns.sort()
    
    def __getitem__(self, index):
        fn = self.fns[index]
        task_str = extract_task(fn)
        task = task_str[0] + 'x' + task_str[1:]
        print(task)
        task = self._task_tokenizer[task]
        task = torch.tensor([task])
        mat = loadmat(os.path.join(self.data_dir, fn))
        action_dict = {}
        action_dict['x0'] = mat['x0']
        action_dict['y0'] = mat['y0']
        action_dict['mask'] = mat['mask']
        action_dict['ATy0'] = mat['ATy0']
        action_dict['gt'] = mat['gt'] 
        action_dict['x0'] = np.clip(action_dict['x0'], a_min=0, a_max = None)
        

        x = mat['x0'][..., 0].reshape(1, 128, 128)
        states = x.reshape(1, -1)
        rtg = (self.rtg_target - EvaluationDataset._min_rtg)/(EvaluationDataset._max_rtg - EvaluationDataset._min_rtg)
        rtg = torch.Tensor([rtg]).reshape(1, 1)
        actions = torch.zeros((self.action_dim))
        return (states, rtg, actions, task), action_dict
    
    
class SpiEvaluationDataset(BaseDataset):
    def __init__(self, block_size, data_dir, action_dim, rtg_target) -> None:
        super().__init__(block_size, data_dir, action_dim)
        self.rtg_target = rtg_target
        self.fns = [im for im in os.listdir(self.data_dir) if im.endswith('.mat')]
        self.fns.sort()
        
    def __getitem__(self, index):
        fn = self.fns[index]
        mat = loadmat(os.path.join(self.data_dir, fn))
        action_dict = {}
        action_dict['output'] = mat['x0']
        action_dict['input'] = mat['x0']
        action_dict['K'] = mat['K']
        action_dict['gt'] = mat['gt']
        action_dict['x0'] = mat['x0']
        task = '4x_10'
        task = self._task_tokenizer[task]
        task = torch.tensor([task])
        x = mat['x0'].reshape(1, 128, 128)
        states = x.reshape(1, -1)
        rtg = (self.rtg_target - EvaluationDataset._min_rtg)/(EvaluationDataset._max_rtg - EvaluationDataset._min_rtg)
        rtg = torch.Tensor([rtg]).reshape(1, 1)
        actions = torch.zeros((self.action_dim))
        return (states, rtg, actions, task), action_dict
        

    

        




        

        

        

        
        