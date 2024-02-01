import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import os
from contextlib import nullcontext


from transformer.decision_transformer import DecisionTransformer, DecisionTransformerConfig
from dataset.training import TrainingDataset

"""
SET PARAMETERS AND HYPERPARAMETERS HERE
"""

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autoscast(device_type = device_type, dtype = ptdtype)


def ddp_setup(rank, world_size):
    """
    Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend = 'nccl', rank = rank, world_size = world_size)
    torch.cuda.set_device(rank)

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size = batch_size,
        pin_memory = True,
        shuffle = False,
        sampler = DistributedSampler(dataset)
    )


class TrainerConfig:
    #optimizatrion parameters as class attributes -> LOOK AT DECISION TRANSFORMER FOR HINTS

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_data_loader : DataLoader,
                 gpu_id: int,
                 save_every: int,
                 compile: bool = False) -> None:
        
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        if compile:
            self.model = torch.compile(model)
        self.train_data_loader = train_data_loader
        self.save_every = save_every
        self.model = DDP(model, device_ids = [gpu_id])


    def _run_batch(self, trajectory):
        self.optimizer.zero_grad()
        states, actions, rtg, traj_masks, task, timesteps, output_masks = trajectory
        actions_target = torch.clone(actions).detach()
        with ctx:
            actions_preds, _ = self.model(actions, rtg, states, timesteps, task, output_masks)
        actions_preds = actions_preds.view(-1, actions_preds.shape[-1])[traj_masks.view(-1, traj_masks.shape[-1]) > 0]
        actions_target = actions_target.view(-1, actions_target.shape[-1])[traj_masks.view(-1, traj_masks.shape[-1]) > 0]
        loss = F.mse_loss(actions_preds, actions_target)
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
        self.optimizer.step()

    
    def _save_checkpoint(self):
        ckp = self.model.module.state_dict()
        PATH = "checkpoints/model.pt"
        torch.save(ckp, PATH)

    
    def _run_epoch(self):
        ### do something with model if DDP
        for trajectory in self.train_data_loader:
            self._run_batch(trajectory)

    def train(self):
        model = self.model
        raw_model = model.module if hasattr(self.model, 'module') else model
        optimizer = model.configure_optimizers(config)
        for epoch in range(self.config.max_parameters):
            self._run_epoch()
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint()


        


model_config = DecisionTransformerConfig()
model = DecisionTransformer(model_config)


#if __name__ == '__main__':

#    trainer.train()
#    trainer = MetaTrainer(batch_size = args.batch_size)




