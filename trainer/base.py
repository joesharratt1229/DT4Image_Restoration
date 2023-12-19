import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import higher
from typing import Tuple, Dict, Optional
import random
from copy import deepcopy
from tqdm import tqdm

from utils.replay import MetaReplayBuffer
from rl.critic.network import Critic
from rl.actor.network import Policy
from env.base import PnPEnv
from solver.solve import AdmmSolver
from dataset.training import TrainingDataset
from utils.gaussian import ContinuousNoise

#grad_clip_val = 1e-3

LOG_FILE_PATH = "training_log.txt"

with open(LOG_FILE_PATH, 'w') as file:
    file.write("Training Log\n")


def clip_gradients(gradients):
    max_norm = 30
    total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g) for g in gradients]))

    clip_coef = max_norm/(total_norm + 1e-6)

    clip_coef_clamped = torch.clamp(clip_coef, max = 1.0)

    for g in gradients:
        if g is not None:
            g.mul_(clip_coef_clamped)

    return [g.detach() for g in gradients]




class MetaTrainer(nn.Module):

    def __init__(self, 
                 batch_size: int) -> None:
        super().__init__()
        #will be 15000
        self._num_train_steps = 15000
        #will be warmup
        self._warmup = 20

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        noise_mod = ContinuousNoise()
            
        self.replay_buffer = MetaReplayBuffer()
        self.critic = Critic(num_outputs = 1, depth = 18).float().to(self.device)
        self.actor = Policy(5, depth = 18).float().to(self.device)
        self.env = PnPEnv(noise_model=noise_mod, solver = AdmmSolver())
        self.solver = AdmmSolver()
        self.dataset = TrainingDataset()
        self.data_loader = iter(DataLoader(self.dataset, 
                                      batch_size = batch_size))
        
        self.loop_penalty = 0.05
        self.discount = 0.99
        self.lambda_e = 0.2
        self.tau = 0.001
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), 0.0005)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 0.001)
        self.max_episode_step = 6
        self._maml_steps = 5
        self._critic_loss = nn.MSELoss()
        self.batch_size = batch_size


    @property
    def _tasks(self):
         return {
              'sampling_strategy': ['cartesian', 'variable_density'],
              'noise_level_range': [5, 20],
              'acceleration_range': range(4, 10, 2),
              'density_range': {'min_density': [0.1, 0.5],
                                'max_density': [0.5, 0.9]}
         }
    
    @property
    def _iterate_data(self) -> Optional[torch.Tensor]:
        try:
            next_batch  = next(self.data_loader)
            return next_batch.to(self.device)
        except StopIteration:
             self.data_loader = iter(DataLoader(self.dataset, batch_size=self.batch_size))
             next_batch = next(self.data_loader)
             return next_batch.to(self.device)
    

    def _soft_update(self, src: nn.Module, target: nn.Module) -> None:
        for param_source, param_target in zip(src.parameters(), target.parameters()):
            param_target.data.copy_(
                param_target.data * (1-self.tau) + param_source.data * self.tau
            )
        
    
    def _sample_replay(self, 
                       task_idx: Optional[int] = None
                       ) -> Tuple:
        if task_idx:
            observations, _ = self.replay_buffer.sample(task_idx)
        else:
            observations, task_idx = self.replay_buffer.sample()
        policy_ob, env_ob = self.env.reset(observations, self.device)
        return (policy_ob, env_ob), task_idx
            
        
    def _sample_tasks(self) -> Tuple[Dict, torch.Tensor]:
        tasks = self._tasks
        sampling_strategy = random.choice(tasks['sampling_strategy'])
        noise_level = np.random.uniform(tasks['noise_level_range'][0], tasks['noise_level_range'][1])
        
        if sampling_strategy == 'density':
            min_density = np.random.uniform(*tasks['density_range']['min_density'])
            max_density = np.random.uniform(*tasks['density_range']['max_density'])
            task_dict = {'mask': sampling_strategy, 'noise_level': noise_level, 
                         'min_density': min_density, 'max_density': max_density}
        else:
            acceleration = random.choice(list(tasks['acceleration_range']))
            task_dict = {'mask': sampling_strategy, 'noise_level': noise_level, 
                          'acceleration': acceleration}
            
        task_dict['index'] = tasks['sampling_strategy'].index(sampling_strategy)
            
        return task_dict, self._iterate_data


    @torch.no_grad()
    def _run_observation(self, observations, task_dict):
        policy_obs, env_obs = observations
        actions, _, _  = self.actor(policy_obs, None, True)
        done = actions['idx_stop']
        
        inputs = env_obs['variables'], env_obs['y0'], env_obs['mask']
        
        next_var = self.solver(inputs, actions)
        new_envs_obs = env_obs
        new_envs_obs['variables'] = next_var
        return env_obs, new_envs_obs, done


    def _save_experience(self, 
                         next_obs: Dict,
                         curr_obs: Dict,
                         task_index: int,
                         curr_episode_step: int) -> None:
        
        if curr_episode_step == 0:
             self.replay_buffer.append(task_index, curr_obs, self.device)

        self.replay_buffer.append(task_index, next_obs, self.device)
        return


    def _run_update(self) -> Tuple[float, float, float]:
        with higher.innerloop_ctx(self.critic, self.critic_optim, copy_initial_weights=False) as (f_critic, diff_critic_optim), \
             higher.innerloop_ctx(self.actor, self.actor_optim, copy_initial_weights=False) as (f_actor, diff_actor_optim):
            policy_loss, val_loss, entropy_loss = 0, 0, 0
            f_actor.train()
            f_critic.train()
            observations, task_index = self._sample_replay()
            

            for episode in range(self._maml_steps):
                if episode > 0:
                    observations, _ = self._sample_replay(task_index)
                policy_ob, env_ob = observations
                actions, action_log_prob, dist_entropy =  f_actor(policy_ob, None, True)
                inputs = env_ob['variables'], env_ob['y0'], env_ob['mask']
                obs2 = self.solver(inputs, actions)
                input_tuple = (obs2, env_ob['y0'])
                reward = self.env.compute_reward(input_tuple)
                val_c = f_critic(policy_ob)
                reward -= self.loop_penalty

                policy_ob2 = self.env.build_state_action_ob(obs2, env_ob)

                with torch.no_grad():
                    V_next_target = self.critic_target(policy_ob2)

                    V_next_target = (self.discount * (1 - actions['idx_stop'].float())).unsqueeze(-1) * V_next_target
                    Q_target = (V_next_target.mT + reward).mT
                
                advantage = (Q_target - val_c).clone().detach()
                adv_loss = action_log_prob * advantage

                V_next = f_critic(policy_ob2)
                V_next = (self.discount * (1 - actions['idx_stop'].float())).unsqueeze(-1) * V_next
                #compute dpg loss _> continous denoising strength and penalty parameters
                ddpg_loss = (V_next.mT + reward).mT
                episode_policy_loss = - (adv_loss + ddpg_loss + self.lambda_e * dist_entropy).mean()
                #f_actor.zero_grad()
                diff_actor_optim.step(episode_policy_loss, grad_callback = lambda x: clip_gradients(x), override={'lr': [0.0003]})

                episode_value_loss = self._critic_loss(Q_target, val_c)
                #f_critic.zero_grad()
                diff_critic_optim.step(episode_value_loss, grad_callback=lambda x: clip_gradients(x), override={'lr': [0.0001]})


                with open(LOG_FILE_PATH, 'a') as file:
                    file.write(f'Inner loop policy loss: {episode_policy_loss} \t Inner loop value loss {episode_value_loss} \t Entropy loss {dist_entropy}')


                self._soft_update(f_critic, self.critic_target)
                env_ob['variables'] = obs2
                done = actions['idx_stop']
                observations, done = self.env.step(env_ob, done)
                policy_loss += episode_policy_loss
                val_loss += episode_value_loss
                entropy_loss += dist_entropy


                if (done.eq(1).all().item() == True) or episode == 2:
                    break
            return policy_loss, val_loss, entropy_loss, task_index


    def train(self):
        task_dict, observations = self._sample_tasks()
        psnrs = []
        observations = self.env.build_init_ob(observations, task_dict)

        for step in tqdm(range(self._num_train_steps)):
            observations, next_obs, done = self._run_observation(observations, task_dict)
            self._save_experience(next_obs, observations, task_dict['index'], self.env.episode_num)
            observations, done = self.env.step(next_obs, done)
            
            if (done.eq(1).all().item() == True) or (self.env.episode_num == self.max_episode_step):
                if (step  > self._warmup) :
                    self.critic_target = deepcopy(self.critic)

                    policy_loss, value_loss, entropy_loss, _ = self._run_update()

                    with open(LOG_FILE_PATH, 'a') as file:
                        file.write(f'{step}\t Policy Loss: {policy_loss} \t Value Loss: {value_loss} \t Entropy loss: {entropy_loss}')

                    self.actor.zero_grad()
                    policy_loss.backward(retain_graph = True)
                    nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 2000)
                    for param in self.actor.parameters():
                        print(param.grad)
                    self.actor_optim.step()

                    #potentially introduce override
                    self.critic.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 2000)
                    self.critic_optim.step()
                

                task_dict, observations = self._sample_tasks()
                observations = self.env.build_init_ob(observations, task_dict)

        torch.save(self.actor.state_dict(), 'model_weights/learned_actor.pth')
        torch.save(self.critic.state_dict(), 'model_weights/learned_critic.pth')