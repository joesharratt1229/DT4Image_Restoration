import collections
from typing import Tuple, Dict, Optional
import numpy as np
import torch

Experience = collections.namedtuple(
    'Experience', field_names = ['variables', 'y0', 'Aty0', 'mask', 'T', 'noise_map']
)

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        #efficient operations -> adding from both ends
        self.buffer = collections.deque(maxlen = capacity)

    def __len__(self) -> int:
        return len(self.buffer)
    
    def append(self, ob: Experience) -> None:
        self.buffer.append(ob)
        return
    
    def sample(self, batch_size: int = 48) -> Dict:
        indices = np.random.choice(range(0,self.__len__()), 
                                   batch_size,
                                   replace = False)
        
        variables, y0, Aty0, mask, T, noise_map = zip(*[self.buffer[idx] for idx in indices])
        variables = torch.stack(variables, dim = 0)
        y0 = torch.stack(y0, dim = 0)
        Aty0 = torch.stack(Aty0, dim = 0)
        mask = torch.stack(mask, dim  = 0)
        T = torch.stack(T, dim = 0)
        noise_map = torch.stack(noise_map, dim = 0)
        env_dict = {'variables': variables, 'y0': y0, 'Aty0': Aty0, 
                      'mask': mask, 'T': T, 'noise_map': noise_map}
        
        return env_dict
    


class MetaReplayBuffer:
    _tasks = ['radial', 'cartesian', 'variable_density']

    def __init__(self) -> None:
        self.buffers = {}
        for task_index in range(len(self._tasks)):
            self.buffers[task_index] = ReplayBuffer(capacity=240)

    def __len__(self) -> int:
        total_length = sum(len(self.buffers[task_index]) for task_index in self.buffers.keys())
        return total_length
    
    def sample(self, 
               task_index: Optional[int] = None
               ) -> Tuple[Dict, int]:
        if task_index:
            observations = self.buffers[task_index].sample()
        else:
            task_index = np.random.randint(0, len(self._tasks))
            try:
                observations = self.buffers[task_index].sample()
            except ValueError as e:
                return self.sample()

        return observations, task_index
    
    def append(self, task_index: int, ob: Dict, device) -> None:
        for k, v in ob.items():
            if device.type != 'cpu':
                ob[k] = ob[v].detach().clone().cpu()
        batch_size = ob['variables'].shape[0]

        for i in range(batch_size):
            experience = Experience(variables = ob['variables'][i], 
                                y0 = ob['y0'][i], 
                                Aty0 = ob['Aty0'][i],
                                mask =  ob['mask'][i], 
                                noise_map = ob['noise_map'][i],
                                T = ob['T'][i])
            self.buffers[task_index].append(experience)
        return 
        


        

 
        
        





    

