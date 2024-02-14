import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

import argparse 
import os
from contextlib import nullcontext
from typing import Optional


from transformer.decision_transformer import DecisionTransformer, DecisionTransformerConfig
from dataset.datasets import TrainingDataset, EvaluationDataset
from evaluation.env import PnPEnv
from evaluation.noise import UNetDenoiser2D

"""
In this implementatiion not going to scale rtgs or rtg targets. If doesnt work properly may look to scale rtg targets between 0 and 1.
"""



train_dict = {
    'learning_rate' : 3e-4,
    'beta' :(0.09, 0.95),
    'weight_decay' : 0.1,
    'grad_norm_clipping': 0.1,
    'num_workers': 0,
}

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


def prepare_dataloader(dataset: Dataset, batch_size: int, ddp: bool):
    if ddp:
        return DataLoader(
            dataset,
            batch_size = batch_size,
            pin_memory = True,
            shuffle = False,
            sampler = DistributedSampler(dataset)
        )
    else:
        return DataLoader(
            dataset, 
            batch_size,
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
                 action_dim: int,
                 train_data_loader : DataLoader,
                 optimizer: torch.optim,
                 save_every: int,
                 env,
                 gpu_id: Optional[int] = None,
                 ddp: bool = False,
                 compile: bool = False) -> None:
        
        self.config = train_config
        self.optimizer = optimizer
        self.ddp = ddp
        if ddp:
            self.gpu_id = gpu_id
            self.model = model.to(self.gpu_id)
            self.model = DDP(model, device_ids = [gpu_id])
        else:
            self.model = model

        # ADD ALL ARGUMENTS FOR VALIDATION DATASET
        self.evaluation = EvaluationDataset(block_size = train_config.block_size, rtg_scale = 1, data_dir='evaluation/image_dir/', action_dim= action_dim, rtg_target = 16)
        if compile:
            self.model = torch.compile(model)

        
        self.train_data_loader = train_data_loader
        self.save_every = save_every
        self.env = env


    def _run_batch(self, trajectory):
        states, actions, rtg, traj_masks, timesteps = trajectory
        if self.ddp:
            states, actions, rtg, traj_masks, timesteps = states.to(self.gpu_id), actions.to(self.gpu_id), rtg.to(self.gpu_id), traj_masks.to(self.gpu_id), timesteps.to(self.gpu_id)

        actions_target = torch.clone(actions).detach()
        with ctx:
            actions_preds, _ = self.model(rtg, states, timesteps, actions)
            actions_preds = actions_preds.view(-1, actions_preds.shape[-1])[traj_masks.view(-1, traj_masks.shape[-1]) > 0]
            actions_target = actions_target.view(-1, actions_target.shape[-1])[traj_masks.view(-1, traj_masks.shape[-1]) > 0]
            loss = F.mse_loss(actions_preds, actions_target)

        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

    @torch.no_grad()
    def _run_evaluation(self):
        #(Batch_size, 1, 3*128*128), (Batch_size, 1, 1), (Batch_size, 1, 1)
        max_step = 30
        policy_inputs, mat = self.evaluation.get_eval_obs(index = 0)
        eval_states, eval_rtg, eval_T, eval_actions = policy_inputs
        if self.ddp:
            eval_states, eval_rtg, eval_T, eval_actions = eval_states.to(self.gpu_id), eval_rtg.to(self.gpu_id), eval_T.to(self.gpu_id), eval_actions.to(self.gpu_id)

        done = False
        states = self.env.reset(mat, self.ddp, self.gpu_id)
        pred_actions, action_dict = self.model(eval_rtg, eval_states, eval_T, actions = None)

        for index in range(1, max_step+1):
            states, reward, done = self.env.step(states, action_dict)
            scaled_rtg = eval_rtg - reward/self.evaluation.rtg_scale

            policy_ob = self.env.get_policy_ob(states)

            if (done) or (index == max_step):
                print(f'Observed reward: {reward}')
                return reward

            eval_actions[:, index] = pred_actions
            eval_states[:, index] = policy_ob
            eval_rtg[:, index] = scaled_rtg
            eval_T[:, index] = index
            pred_actions, action_dict = self.model(eval_rtg, eval_states, eval_T, eval_actions)


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
                self._run_evaluation()


def main(rank, save_every, ddp, world_size, compile_arg):
    if ddp:
        ddp_setup(rank, world_size)
    data_loader = prepare_dataloader(dataset, train_config.batch_size, ddp)
    trainer = Trainer(model, train_config, model_config.action_dim, data_loader, optimizer,save_every, env, rank = rank, ddp = ddp,compile = compile_arg)
    trainer.train()
    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for decision transformer')
    parser.add_argument('--batch_size', type = int, required = True)
    parser.add_argument('--block_size', type = int, required = True)
    parser.add_argument('--ddp', type = bool, required = True)
    parser.add_argument('--compile', type = bool, required = True)
    parser.add_argument('--save_every', type = int, required = True)
    parser.add_argument('--max_epochs', type = int, required = True)
    args = parser.parse_args()

    train_dict['batch_size'] = args.batch_size
    train_dict['block_size'] = args.block_size
    train_dict['max_epochs'] = args.max_epochs
    denoiser = UNetDenoiser2D(ckpt_path='evaluation/pretrained/unet-nm.pt')
    train_config = TrainerConfig(**train_dict)
    model_config = DecisionTransformerConfig(block_size = train_config.block_size)
    model = DecisionTransformer(model_config)
    optimizer = model.configure_optimizers(train_config)
    #ADD NECESSARY ARGUMENTS FOR TRAIN DATASET
    dataset = TrainingDataset(block_size = train_config.block_size, 
                              rtg_scale= 1, 
                              data_dir='dataset/data/data_dir', 
                              action_dim = model_config.action_dim, 
                              state_file_path='dataset/data/state_dir/data.h5')
    
    env = PnPEnv(max_episode_step=30, denoiser = denoiser)
    data_loader = prepare_dataloader(dataset, train_config.batch_size, args.ddp)

    if args.ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(main, args = {args.save_every, args.ddp, world_size, args.compile}, nprocs=world_size)
    
    else: 
        main(rank = None, save_every=args.save_every, ddp = False, world_size = None, compile_arg = False)






#SPAWN MULTIPLE PROCESSES

#if __name__ == '__main__':

#    trainer.train()
#    trainer = MetaTrainer(batch_size = args.batch_size)

