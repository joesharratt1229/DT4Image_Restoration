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
    'beta' :(0.9, 0.95),
    'weight_decay' : 0.1,
    'grad_norm_clipping': 1.0,
    'num_workers': 0,
}

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type = device_type, dtype = ptdtype)
        

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
    if ddp == True:
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
            pin_memory = True
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
                 max_timesteps,
                 context_length,
                 train_data_loader : DataLoader,
                 optimizer: torch.optim,
                 save_every: int,
                 env,
                 eval_loader,
                 gpu_id,
                 ddp: bool = False,
                 compile: bool = False) -> None:
        
        self.config = train_config
        self.optimizer = optimizer
        self.action_dim = action_dim
        self.max_timesteps = max_timesteps
        self.context_length = context_length
        self.ddp = ddp
        if ddp:
            self.gpu_id = gpu_id
            self.model = model.to(gpu_id)
            self.model = DDP(model, device_ids = [self.gpu_id])
        else:
            self.model = model.to(device_type)
            self.gpu_id = None

        # ADD ALL ARGUMENTS FOR VALIDATION DATASET)
        if compile:
            self.model = torch.compile(model)

        
        self.train_data_loader = train_data_loader
        self.eval_loader = eval_loader
        self.save_every = save_every
        self.env = env

    @staticmethod
    def _get_latest_action(action_dict, actions_preds, index):
        if index>=3:
            slice_index = -1
        else:
            slice_index = index
        
        actions_preds = actions_preds[0][slice_index]
        
        action_dict['T'] = action_dict['T'][0][slice_index]
        action_dict['mu'] = action_dict['mu'][0][slice_index]
        action_dict['sigma_d'] = action_dict['sigma_d'][0][slice_index]
        return action_dict, actions_preds


    def _run_batch(self, trajectory):
        states, actions, rtg, traj_masks, timesteps = trajectory
        if self.ddp:
            states, actions, rtg, traj_masks, timesteps = states.to(self.gpu_id), actions.to(self.gpu_id), rtg.to(self.gpu_id), traj_masks.to(self.gpu_id), timesteps.to(self.gpu_id)
        else:
            states, actions, rtg, traj_masks, timesteps = states.to(device_type), actions.to(device_type), rtg.to(device_type), traj_masks.to(device_type), timesteps.to(device_type)
        actions_target = torch.clone(actions).detach()
        
        with ctx:
            actions_preds, _ = self.model(rtg, states, timesteps, actions)
            traj_masks = traj_masks.expand_as(actions_target)
            actions_preds = actions_preds.view(-1, actions_preds.shape[-1])[traj_masks.view(-1, traj_masks.shape[-1]) > 0]
            actions_target = actions_target.view(-1, actions_target.shape[-1])[traj_masks.view(-1, traj_masks.shape[-1]) > 0]
            loss = F.mse_loss(actions_preds, actions_target)

        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        print('Loss: ' ,{loss})

    def run_evaluation(self, rtg_scale):
        #(Batch_size, 1, 3*128*128), (Batch_size, 1, 1), (Batch_size, 1, 1)
        model_weights = torch.load('final_mod.pt', map_location=torch.device('cpu'))
        self.model.load_state_dict(model_weights)
        self.model.eval()
        
        max_step = 30
        for data in self.eval_loader:
            policy_inputs, mat = data
            states, rtg, _ = policy_inputs
           
            if self.ddp:
                states, rtg = states.to(self.gpu_id), rtg.to(self.gpu_id)
            else:
                states, rtg = states.to(device_type), rtg.to(device_type)

            eval_actions = torch.zeros((1, self.max_timesteps, self.action_dim))
            eval_states = torch.zeros((1, self.max_timesteps, 3*128*128))
            eval_rtg = torch.zeros((1, self.max_timesteps, 1))

            eval_timesteps = torch.arange(start = 0, end=self.max_timesteps).reshape(1, self.max_timesteps, 1).contiguous()
            
            eval_states[0, 0] = states
            eval_rtg[0, 0] = rtg

            done = False
            states = self.env.reset(mat, self.ddp, self.gpu_id)
            old_reward = self.env.compute_reward(states['x'].real.squeeze(dim = 0), states['gt'])
            print('Original reward', old_reward)
            
            pred_actions, action_dict = self.model(eval_rtg[:, :self.context_length], eval_states[:, :self.context_length], eval_timesteps[:, :self.context_length], actions = None)
            if self.context_length > 1:
                action_dict = self._get_latest_action(action_dict, pred_actions, index=0)

            for time in range(1, max_step+1):
                states, reward, done = self.env.step(states, action_dict)
                rtg = reward - old_reward
                old_reward = reward
                print(reward)
                scaled_rtg = eval_rtg[0, time - 1] - rtg/rtg_scale
                policy_ob = self.env.get_policy_ob(states)

                if (done) or (time == max_step):
                    print('Final reward', {reward})
                    break

                eval_actions[:, time - 1] = pred_actions
                eval_states[:, time] = policy_ob
                eval_rtg[:, time] = scaled_rtg

                if time < self.context_length:
                    pred_actions, action_dict = self.model(eval_rtg[:, :1], eval_states[:, :1], eval_timesteps[:, :1], eval_actions[:, :1])
                else:
                    pred_actions, action_dict = self.model(eval_rtg[:,time-self.context_length:time], 
                                                           eval_states[:, time-self.context_length:time], 
                                                           eval_timesteps[:, time-self.context_length:time],
                                                           eval_actions[:, time-self.context_length:time])
                if self.context_length > 1:
                    action_dict, pred_actions = self._get_latest_action(action_dict, pred_actions, index = time)


    def _save_checkpoint(self):
        model = self.model.module if self.ddp else self.model
        ckp = model.state_dict()
        PATH = "checkpoints/model.pt"
        torch.save(ckp, PATH)
    
    def _run_epoch(self):
        ### do somethiisplang with model if DDP
        for trajectory in self.train_data_loader:
            self._run_batch(trajectory)

    def train(self):
        for epoch in range(self.config.max_epochs):
            self._run_epoch()
            if epoch % self.save_every == 0:
                if (self.ddp):
                    try:
                        if (self.gpu_id == 0):
                            self._save_checkpoint()
                            self._run_evaluation()
                    except Exception as e:
                        print('Unknown errror')
                else:
                    self._save_checkpoint()
                    try:
                        self._run_evaluation()
                    except Exception as e:
                        print(f"An error occurred during evaluation")
                    


def main(rank, save_every, ddp, world_size, compile_arg, 
         batch_size, block_size, max_epochs):
    if ddp:
        ddp_setup(rank, world_size)
    
    train_dict['batch_size'] = batch_size
    train_dict['block_size'] = block_size
    train_dict['max_epochs'] = max_epochs
    
    denoiser = UNetDenoiser2D(ckpt_path='evaluation/pretrained/unet-nm.pt')
    train_config = TrainerConfig(**train_dict)
    model_config = DecisionTransformerConfig(block_size = train_config.block_size)
    model = DecisionTransformer(model_config)
    optimizer = model.configure_optimizers(train_config)
    #ADD NECESSARY ARGUMENTS FOR TRAIN DATASET
    env = PnPEnv(max_episode_step=30, denoiser = denoiser)
    dataset = TrainingDataset(block_size = train_config.block_size//3, 
                              rtg_scale= 1, 
                              data_dir='dataset/data/data_dir/CSMRI', 
                              action_dim = model_config.action_dim, 
                              state_file_path='dataset/data/state_dir/data.h5')
    
    eval_dataset = EvaluationDataset(block_size = train_config.block_size//3, rtg_scale = 1, data_dir='evaluation/image_dir/', action_dim= 3, rtg_target = 16)
    eval_loader = DataLoader(dataset = eval_dataset, batch_size=1)
    
    data_loader = prepare_dataloader(dataset, train_config.batch_size, ddp)
    trainer = Trainer(model, 
                      train_config, 
                      model_config.action_dim,
                      model_config.max_timestep,
                      train_config.block_size,
                      data_loader, 
                      optimizer,save_every, 
                      env,
                      eval_loader,
                      rank,
                      ddp = ddp,
                      compile = compile_arg)
    trainer.train()
    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for decision transformer')
    parser.add_argument('--batch_size', type = int, required = True)
    parser.add_argument('--block_size', type = int, required = True)
    parser.add_argument('--ddp', action='store_true', help='Enable distributed data parallel')
    parser.add_argument('--compile', action='store_true', help='Enable compilation')
    parser.add_argument('--save_every', type = int, required = True)
    parser.add_argument('--max_epochs', type = int, required = True)
    args = parser.parse_args()
    if args.ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(args.save_every, args.ddp, world_size, 
                             args.compile, args.batch_size, args.block_size, 
                             args.max_epochs), nprocs=world_size)
    else: 
        main(rank = None, save_every=args.save_every, 
             ddp = False, world_size = None, compile_arg = False,
             batch_size=args.batch_size, block_size=args.block_size, 
             max_epochs=args.max_epochs)






#SPAWN MULTIPLE PROCESSES

#if __name__ == '__main__':

#    trainer.train()
#    trainer = MetaTrainer(batch_size = args.batch_size)

