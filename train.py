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
from dataset.datasets import TrainingDataset

"""
SET PARAMETERS AND HYPERPARAMETERS HERE
"""

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
                 train_config,
                 train_data_loader : DataLoader,
                 optimizer: torch.optim,
                 gpu_id: int,
                 save_every: int,
                 compile: bool = False) -> None:
        
        self.config = train_config
        
        self.gpu_id = gpu_id
        self.optimizer = optimizer
        self.model = model.to(gpu_id)
        if compile:
            self.model = torch.compile(model)
        self.train_data_loader = train_data_loader
        self.save_every = save_every
        self.model = DDP(model, device_ids = [gpu_id])


    def _run_batch(self, trajectory):
        states, actions, rtg, traj_masks, timesteps= trajectory
        actions_target = torch.clone(actions).detach()
        with ctx:
            actions_preds, _ = self.model(actions, rtg, states, timesteps)
            actions_preds = actions_preds.view(-1, actions_preds.shape[-1])[traj_masks.view(-1, traj_masks.shape[-1]) > 0]
            actions_target = actions_target.view(-1, actions_target.shape[-1])[traj_masks.view(-1, traj_masks.shape[-1]) > 0]
            loss = F.mse_loss(actions_preds, actions_target)
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

    
    def _save_checkpoint(self):
        raw_model = self.model.module if model.module else model
        ckp = raw_model.module.state_dict()
        PATH = "checkpoints/model.pt"
        torch.save(ckp, PATH)

    
    def _run_epoch(self):
        ### do somethiisplang with model if DDP
        for trajectory in self.train_data_loader:
            self._run_batch(trajectory)

    def train(self):
        for epoch in range(self.config.max_epochs):
            self._run_epoch()
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint()


train_dict = {
    'learning_rate' : 3e-4,
    'beta' :(0.09, 0.95),
    'weight_decay' : 0.1,
    'max_epochs': 10,
    'grad_norm_clipping': 0.1,
    'num_workers': 0,
    'batch_size': 16,
    'block_size': 4
}

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autoscast(device_type = device_type, dtype = ptdtype)
        

train_config = TrainerConfig(**train_dict)
model_config = DecisionTransformerConfig(block_size = train_config.block_size)
model = DecisionTransformer(model_config)
raw_model = model.module if hasattr(model, 'module') else model
optimizer = model.configure_optimizers(train_config)
dataset = TrainingDataset(block_size = train_config.block_size)

data_loader = prepare_dataloader(dataset, train_config.batch_size)

def main(rank: int, world_size: int, save_every: int):
    ddp_setup(rank, world_size)

#SPAWN MULTIPLE PROCESSES

#if __name__ == '__main__':

#    trainer.train()
#    trainer = MetaTrainer(batch_size = args.batch_size)

