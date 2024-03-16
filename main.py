import os
import argparse
import logging
from contextlib import nullcontext

import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp


from transformer.decision_transformer import DecisionTransformer, DecisionTransformerConfig
from train import Trainer, TrainerConfig
from evaluation.eval import Evaluator
from evaluation.noise import UNetDenoiser2D
from evaluation.env import PnPEnv
from dataset.datasets import TrainingDataset

PRETRAINED_MODEL_PATH = 'checkpoints/model_2.pt' 

logging.basicConfig(filename='outputs.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

train_dict = {
    'learning_rate' : 3e-4,
    'beta' :(0.9, 0.95),
    'weight_decay' : 0.1,
    'grad_norm_clipping': 1.0,
    'num_workers': 0,
    'lr_decay': True
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




def train_model(rank, save_every, ddp, world_size, compile_arg, 
                batch_size, block_size, max_epochs):
    if ddp:
        ddp_setup(rank, world_size)
    
    train_dict['batch_size'] = batch_size
    train_dict['block_size'] = block_size
    train_dict['max_epochs'] = max_epochs
    
    #denoiser = UNetDenoiser2D(ckpt_path='evaluation/pretrained/unet-nm.pt')
    train_config = TrainerConfig(**train_dict)
    model_config = DecisionTransformerConfig(block_size = train_config.block_size)
    model = DecisionTransformer(model_config)
    optimizer = model.configure_optimizers(train_config)
    #ADD NECESSARY ARGUMENTS FOR TRAIN DATASET
    #env = PnPEnv(max_episode_step=30, denoiser = denoiser, device_type = device_type)
    dataset = TrainingDataset(block_size = train_config.block_size//3, 
                              data_dir='dataset/data/data_dir/csmri', 
                              action_dim = model_config.action_dim, 
                              state_file_path='dataset/data/state_dir/data_1.h5')
    
    
    dataset_length = dataset.__len__()
    max_steps = int(dataset_length//train_dict['batch_size']) * train_dict['max_epochs']
    print(max_steps)
    print(dataset_length)
    
    data_loader = prepare_dataloader(dataset, train_config.batch_size, ddp)
    trainer = Trainer(model, 
                      train_config, 
                      data_loader, 
                      optimizer,
                      save_every, 
                      max_steps,
                      rank,
                      ctx,
                      device_type,
                      ddp = ddp,
                      compile = compile_arg)
    trainer.train()
    if ddp:
        destroy_process_group()
        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for decision transformer - train and evaluation')
    parser.add_argument('--block_size', type = int, required = True)
    subparsers = parser.add_subparsers(dest = 'mode', help = 'Modes: train or evaluation')
    train_parser = subparsers.add_parser('train') 
    train_parser.add_argument('--batch_size', type = int, required = True)
    train_parser.add_argument('--ddp', action='store_true', help='Enable distributed data parallel')
    train_parser.add_argument('--compile', action='store_true', help='Enable compilation')
    train_parser.add_argument('--save_every', type = int, required = True)
    train_parser.add_argument('--max_epochs', type = int, required = True)
    
    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('--rtg', help = 'Desired rtg')
    eval_parser.add_argument('--max_timesteps', help = 'Timesteps')
    

    args = parser.parse_args()
    if args.mode == 'train':
        if args.ddp:
            world_size = torch.cuda.device_count()
            mp.spawn(train_model, args=(args.save_every, args.ddp, world_size, 
                                args.compile, args.batch_size, args.block_size, 
                                args.max_epochs), nprocs=world_size)
        else: 
            train_model(rank = None, save_every=args.save_every, 
                        ddp = False, world_size = None, compile_arg = False,
                        batch_size=args.batch_size, block_size=args.block_size, 
                        max_epochs=args.max_epochs)
            
    else:
        model_config = DecisionTransformerConfig(block_size = args.block_size)
        model = DecisionTransformer(model_config)
        model = model.to(device_type)
        denoiser = UNetDenoiser2D(ckpt_path='evaluation/pretrained/unet-nm.pt')
        env = PnPEnv(max_episode_step=30, denoiser = denoiser, device_type = device_type)

        evaluate = Evaluator(model = model, model_path = PRETRAINED_MODEL_PATH, action_dim = 3, 
                             max_timesteps=30, env = env, compile = False, device_type=device_type, 
                             block_size=args.block_size, rtg_target = args.rtg)
        dataset_paths = ['evaluation/image_dir/vanilla/4_5/', 'evaluation/image_dir/vanilla/4_15/', 'evaluation/image_dir/vanilla/4_10/',
                         'evaluation/image_dir/vanilla/2_15/', 'evaluation/image_dir/vanilla/2_10/', 'evaluation/image_dir/vanilla/2_5/',
                         'evaluation/image_dir/vanilla/8_15/', 'evaluation/image_dir/vanilla/8_10/', 'evaluation/image_dir/vanilla/8_5/']
        
        
        #evaluate.run(dataset_paths)
        evaluate.run(dataset_paths)

        