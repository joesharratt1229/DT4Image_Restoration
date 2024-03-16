import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

import argparse 
import os
from contextlib import nullcontext
import time
import math
from typing import Optional

import wandb
import logging


from transformer.decision_transformer import DecisionTransformer, DecisionTransformerConfig
from dataset.datasets import TrainingDataset, EvaluationDataset
from evaluation.env import PnPEnv
from evaluation.noise import UNetDenoiser2D

"""
In this implementatiion not going to scale rtgs or rtg targets. If doesnt work properly may look to scale rtg targets between 0 and 1.
"""







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
                 save_every: int,
                 max_steps,
                 gpu_id,
                 ctx,
                 device_type,
                 ddp: bool = False,
                 compile: bool = False) -> None:
        
        self.config = train_config
        self.optimizer = optimizer
        self.ddp = ddp
        self.device_type = device_type
        if ddp:
            self.gpu_id = gpu_id
            self.model = model.to(gpu_id)
            self.model = DDP(model, device_ids = [self.gpu_id])
        else:
            self.model = model.to(self.device_type)
            self.gpu_id = None

        # ADD ALL ARGUMENTS FOR VALIDATION DATASET)
        if compile:
            self.model = torch.compile(model)

        
        self.train_data_loader = train_data_loader
        self.save_every = save_every
        self.max_steps = max_steps
        self.warmup_steps = 1250
        self.current_step = 0
        self.ctx = ctx

    
    def _increment_step(self):
        self.current_step += 1


    def _run_batch(self, trajectory):
        states, actions, rtg, traj_masks, timesteps, task = trajectory
        if self.ddp:
            states, actions, rtg, traj_masks, timesteps, task = states.to(self.gpu_id), actions.to(self.gpu_id), rtg.to(self.gpu_id), traj_masks.to(self.gpu_id), timesteps.to(self.gpu_id), task.to(self.gpu_id)
        else:
            states, actions, rtg, traj_masks, timesteps, task = states.to(self.device_type), actions.to(self.device_type), rtg.to(self.device_type), traj_masks.to(self.device_type), timesteps.to(self.device_type), task.to(self.device_type)
        actions_target = torch.clone(actions).detach()
        rtg_target = torch.clone(rtg).detach()
        targets = torch.cat([actions_target, rtg_target], dim = -1)
        

        
        with self.ctx:
                preds, _ = self.model(rtg, states, timesteps, task, actions)
                traj_masks = traj_masks.expand_as(targets)
                preds = preds.view(-1, preds.shape[-1])[traj_masks.view(-1, traj_masks.shape[-1]) > 0]
                targets = targets.view(-1, targets.shape[-1])[traj_masks.view(-1, traj_masks.shape[-1]) > 0]
                loss = F.mse_loss(preds, targets)
        
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        
        

        self._increment_step()
        wandb.log({"loss": loss})
        
        #warmup tokens
        if self.current_step < self.warmup_steps:
            lr_mult = self.current_step/self.warmup_steps
            lr = self.config.learning_rate * lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print(lr)
        #cosine decya
        else:
            progress = float(self.current_step)/float(self.max_steps)
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.config.learning_rate * lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                    

    def _save_checkpoint(self, epoch):
        model = self.model.module if self.ddp else self.model
        ckp = model.state_dict()
        PATH = f"checkpoints/model_{epoch}.pt"
        torch.save(ckp, PATH)
        

    def _run_epoch(self):
        ### do somethiisplang with model if DDP
        for trajectory in self.train_data_loader:
            self._run_batch(trajectory)

    def train(self):
        
        wandb.login(key='d26ee755e0ba08a9aff87c98d0cedbe8b060484b')
        wandb.init(project='rtg_pred', entity='joesharratt1229')
        wandb.watch(self.model)
        start_time = time.time()
        for epoch in range(self.config.max_epochs):
            self._run_epoch()
            logging.debug(f'Epoch {epoch}')
            if epoch % self.save_every == 0:
                if (self.ddp):
                    try:
                        if (self.gpu_id == 0):
                            self._save_checkpoint(epoch)
                            #self.run_evaluation()
                    except Exception as e:
                        print('Unknown errror')
                else:
                    self._save_checkpoint(epoch)
                    #try:
                        #self.run_evaluation()
                    #except Exception as e:
                    #    print(f"An error occurred during evaluation")
            
            
            end_time = time.time()
            time_duration = start_time - end_time
            wandb.log({"training_duration": time_duration})
                   
        wandb.finish()