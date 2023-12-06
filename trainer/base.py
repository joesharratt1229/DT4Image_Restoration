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
        self.critic = Critic(num_outputs = 1, depth = 18).to(self.device)
        #self.critic_target = Critic(num_outputs=1, depth = 18).to(self.device)
        self.actor = Policy(5, depth = 18).to(self.device)
        self.env = PnPEnv(noise_model=noise_mod, solver = AdmmSolver())
        self.solver = AdmmSolver()
        self.dataset = TrainingDataset()
        self.data_loader = iter(DataLoader(self.dataset, 
                                      batch_size = batch_size))
        
        self.loop_penalty = 0.05
        self.discount = 0.99
        self.lambda_e = 0.2
        self.tau = 0.001
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), 0.0003)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 0.0001)
        self.max_episode_step = 6
        self._maml_steps = 5

    @property
    def _tasks(self):
         return {
              'sampling_strategy': ['radial', 'cartesian', 'variable_density'],
              'noise_level_range': [5, 20],
              'acceleration_range': range(2, 10, 2),
              'density_range': {'min_density': [0.1, 0.5],
                                'max_density': [0.5, 0.9]}
         }
    
    @property
    def _iterate_data(self) -> Optional[torch.Tensor]:
        try:
            next_batch  = next(self.data_loader)
            return next_batch.to(self.device)
        except StopIteration:
             return None
    

    def _soft_update(self, src: nn.Module, target: nn.Module) -> None:
        for param_source, param_target in zip(src.parameters(), target.parameters()):
            assert param_source[0] == param_target[0]
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
        policy_ob, env_ob = self.env.reset(observations)
        policy_ob, env_ob = policy_ob.to(self.device), env_ob.to(self.device)
        return (policy_ob, env_ob), task_idx
            
        
    def _sample_tasks(self
                      ) -> Tuple[Dict, torch.Tensor]:
        tasks = self._tasks
        sampling_strategy = random.choice(tasks['sampling_strategy'])
        noise_level = np.random.uniform(tasks['noise_level_range'][0], tasks['noise_level_range'][1])
        
        if sampling_strategy == 'density':
            min_density = np.random.uniform(tasks['density_range']['min_density'])
            max_density = np.random.uniform(tasks['density_range']['max_density'])
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
        new_envs_obs['T'] += + 1/self.env.max_episode_step
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

        print(self.replay_buffer.__len__())
        return

    
    def _run_policy(self, 
                    observations,
                    f_actor, 
                    f_critic, 
                    diff_critic_optim, 
                    diff_actor_optim,
                    inner):
        policy_ob, env_ob = observations
        actions, action_log_prob, dist_entropy =  f_actor(policy_ob)
        reward, obs2 = self.solver(env_ob, actions)
        val_c = f_critic(policy_ob)
        reward -= self.loop_penalty

        with torch.no_grad():
            V_next_target = self.critic_target(obs2)
            V_next_target = (
            self.discount * (1 - actions['idx_stop'].float())).unsqueeze(-1) * V_next_target
            Q_target = V_next_target + reward
            
        advantage = (Q_target - val_c).clone().detach()
        adv_loss = action_log_prob * advantage

        V_next = (self.discount * (1 - actions['idx_stop'].float())).unsqueeze(-1) * V_next
        #compute dpg loss _> continous denoising strength and penalty parameters
        ddpg_loss = V_next + reward

        policy_loss = - (adv_loss + ddpg_loss + self.lambd * dist_entropy)
        
        #advantage or policy loss for this
        #if inner:
        #     adv2_loss = (advantage_prediction - advantage.unsqueeze(-1))*2
        #     policy_loss += adv2_loss
        
        
        policy_loss.mean()
        f_actor.zero_grad()
        nn.utils.clip_grad(f_actor.parameters(), 50)
        diff_actor_optim.step(policy_loss)
        
        critic_loss = nn.MSELoss(Q_target, val_c)
        f_critic.zero_grad()
        nn.utils.clip_grad(f_critic.parameters(), 50)
        diff_critic_optim.step(critic_loss)

        self._soft_update(f_critic, self.critic_target)

        return policy_loss, critic_loss, dist_entropy

    def _run_update(self, 
                    observations: torch.Tensor, 
                    f_actor: nn.Module,
                    f_critic: nn.Module,
                    diff_actor_optim: torch.optim,
                    diff_critic_optim: torch.optim,
                    inner: bool = True
                    ) -> Tuple[float, float, float]:
            
            #observations = self.env.build_ob(observations, task_dict, init = True)
            policy_loss, val_loss, entropy_loss = 0, 0, 0
            
            for episode in self.max_episode_step:
                    episode_policy_loss, episode_value_loss, episode_entropy_loss = self._run_policy(observations, f_actor, f_critic, diff_critic_optim, diff_actor_optim, inner)
                    observations = self.env.step()
                    policy_loss += episode_policy_loss
                    val_loss += episode_value_loss
                    entropy_loss += episode_entropy_loss
                
            return policy_loss, val_loss, entropy_loss


    def train(self):
        task_dict, observations = self._sample_tasks()
        psnrs = []
        observations = self.env.build_init_ob(observations, task_dict)

        for step in tqdm(range(self._num_train_steps)):
            print(step)
            observations, next_obs, done = self._run_observation(observations, task_dict)
            self._save_experience(next_obs, observations, task_dict['index'], self.env.episode_num)
            observations = next_obs
            observations = self.env.step(next_obs)
            
            if (done.eq(1).all().item() == True) or (self.env.episode_num == self.max_episode_step):
                if (step  > self._warmup) :
                    self.critic_target = deepcopy(self.critic)

                    #potentially introduce override
                    with higher.innerloop_ctx(self.critic, self.critic_optim, copy_initial_weights = False) as (f_critic, diff_critic_optim), \
                         higher.innerloop_ctx(self.actor, self.actor_optim, copy_initial_weights = False) as (f_actor, diff_actor_optim):
                        for i in range(self._maml_steps):
                            inner_obs, task_index = self._sample_replay()
                            policy_loss, value_loss, entropy_loss = self._run_update(self, inner_obs, f_actor, f_critic, diff_actor_optim, diff_critic_optim)
                            print(policy_loss)
                            print(value_loss)
                            return
                            
                    batch_obs, _ = self._sample_replay(task_index)
                        
                    meta_policy_loss, meta_value_loss, entropy_loss = self._run_update(self, batch_obs, f_actor, f_critic, diff_actor_optim, diff_critic_optim, batch_obs, inner = False)

                    self.actor.zero_grad()
                    meta_policy_loss.backward()
                    self.actor_optim.step()
                    
                    self.critic.zero_grad()
                    meta_value_loss.backward()
                    self.critic_optim.step()
                    
                task_dict, observations = self._sample_tasks()
                observations = self.env.build_init_ob(observations, task_dict)
            
            

                



                





            


    
