import os
import torch
import torch.utils.data.dataset as dataset
import json
import numpy as np
from PIL import Image


TRAINING_DIR = os.path.join(os.getcwd(), 'dataset/data/Images_128')
TRAINING_DICTIONARY_DIR = ''
STATE_DIR = ''

class TrainingDataset(dataset.Dataset):
    parameters = ['sigma_d', 'mu', 'tau', 'T']

    def __init__(self, block_size) -> None:
        super(TrainingDataset, self).__init__()
        
        self.block_size = block_size
        self._training_dictionary_dir = TRAINING_DICTIONARY_DIR
        self._state_directory = STATE_DIR

        
    def __len__(self):
        return len(os.listdir(self._training_dir))
    
    @staticmethod
    def _get_states(index, traj_start, traj_end, pad = None):
        state_tensors = []

        for trajectory in range(traj_start, traj_end):
            x = torch.from_numpy(np.array(Image.open(f'{STATE_DIR}/image_{index}_x_{trajectory}.png'))).reshape(128, 128)
            z = torch.from_numpy(np.array(Image.open(f'{STATE_DIR}/image_{index}_z_{trajectory}.png'))).reshape(128, 128)
            u = torch.from_numpy(np.array(Image.open(f'{STATE_DIR}/image_{index}_u_{trajectory}.png'))).reshape(128, 128)

            variable = torch.stack([x, z, u])
            state_tensors.append(variable)

        states = torch.stack(state_tensors)

        if pad is not None:
            states = torch.cat([states, torch.zeros(([pad] + list(x.shape[1:])), dtype = x.dtype)])

        return states
    
    @staticmethod
    def _get_actions(self, action_dict, traj_start, traj_end, task, pad = None):
        action_lis = [torch.Tensor(action_dict[key]) for key in action_dict.keys()]
        actions = torch.stack((action_lis), dim = 1)
        
        if pad is not None:
            padding = torch.zeros((pad, actions.shape[1]))
            actions = torch.cat([actions, padding], dim = 0)

        return actions
        


    def __getitem__(self, 
                    index: int
                    ) -> torch.Tensor:
        #TODO tokenizer for the task

        #actions need to be normalised to be between 0 and 1

        traj_name = self._training_dictionary_dir[index]
        traj_path = os.path.join(self._training_dictionary_dir, traj_name)
        
        with open(traj_path, 'r') as file:
            traj_dict = json.loads(file)
        
        traj_len = len(traj_dict['state_path'])
        traj_dict['Actions']['T'] = [value if value % 5 == 0 else 0 for value in traj_dict['Actions']['T']]

        if traj_len >= self.block_size:
            start = np.random.randint(0, traj_len - self.block_size)
            get = lambda x: torch.from_numpy(traj_dict[x][start: start + self.block_size])

            actions = get('actions')
            rtg = torch.from_numpy(traj_dict['rtg'][start: start + self.block_size]).reshape(-1, 1)
            timesteps = torch.arange(start, start + self.block_size)
            task = traj_dict['task']
            states = self._get_states(index, start, start + self.block_size)
            traj_masks = torch.ones(self.block_size, dtype = torch.long)
        else:
            padding_len = self.block_size - traj_len
            concat_pad = lambda x: torch.cat([x, 
                                              torch.zeros(([padding_len] + 
                                                           list(x.shapes[1:])),
                                              dtype = x.dtype)], dim = 0)
            actions = torch.from_numpy(traj_dict['actions'])
            actions = concat_pad(actions)
            rtg = torch.from_numpy(traj_dict['rtg'])
            rtg = concat_pad(rtg)
            task = traj_dict['task']
            traj_masks = torch.cat([torch.ones(traj_len, dtype = torch.long),
                                    torch.zeros(padding_len, dtype = torch.long)],
                                    dim = 0)
            states = self._get_states(index, 0, actions.shape[0], pad = padding_len)
            timesteps = torch.arange(start = 0, end = self.block_size)
            
        out_masks = traj_dict['output_mask']
        output_masks = torch.cat([torch.from_numpy(out_masks[key]) for key in self.parameters], dim = -1)


        return states, actions, rtg, traj_masks, task, timesteps, output_masks








        

        

        

        
        