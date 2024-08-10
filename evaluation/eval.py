import torch
from torch.utils.data import DataLoader
import numpy as np

from evaluation.utils.transformations import calculate_ssim
from PIL import Image

from dataset.datasets import EvaluationDataset

class Evaluator:
    def __init__(self,
                 model,
                 model_path,
                 action_dim,
                 max_timesteps,
                 env,
                 compile,
                 device_type,
                 block_size, 
                 rtg_target) -> None:
        
        model_weights = torch.load(model_path, map_location=device_type)
        
        if compile:
            self.model = torch.compile(model)
        else:
            self.model = model
            
        self.model.load_state_dict(model_weights)
            
            
        self.action_dim = action_dim
        self.max_timesteps = max_timesteps
        self.env = env
        self.device_type = device_type
        self.context_length = block_size//3
        self.rtg_target = rtg_target
        

    def _get_latest_action(self, action_dict, actions_preds, index):
        if index>=self.context_length:
            slice_index = -1
        else:
            slice_index = index
        
        actions_preds = actions_preds[0][slice_index]
        
        action_dict['T'] = action_dict['T'][0][slice_index]
        action_dict['mu'] = action_dict['mu'][0][slice_index]
        action_dict['sigma_d'] = action_dict['sigma_d'][0][slice_index]
        return action_dict, actions_preds
    
    
    def _get_latest_rtg(self, rtg_preds, index):
        if index > self.context_length:
            slice_index = -1
        else:
            slice_index = index
        
        rtg_preds = rtg_preds[0][slice_index -1]
        return rtg_preds
    
    def get_initial_policy_setup(self, policy_inputs, mat):
        states, rtg, _, task = policy_inputs
        states, rtg = states.to(self.device_type), rtg.to(self.device_type)
        eval_actions = torch.zeros((1, self.max_timesteps, self.action_dim)).to(self.device_type)
        eval_states = torch.zeros((1, self.max_timesteps, 1*128*128)).to(self.device_type)
        eval_rtg = torch.zeros((1, self.max_timesteps, 1)).to(self.device_type)

        eval_timesteps = torch.arange(start = 0, end=self.max_timesteps).reshape(1, self.max_timesteps, 1).contiguous().to(self.device_type)
        eval_task = task.repeat(1, self.max_timesteps).to(self.device_type)
        
        eval_states[0, 0] = states
        eval_rtg[0, 0] = rtg

        states = self.env.reset(mat, self.device_type)
        #print('Original reward', old_reward)
        
        with torch.no_grad():
            self.model.eval()
            pred_actions, action_dict = self.model(eval_rtg[:, :self.context_length], 
                                                eval_states[:, :self.context_length], 
                                                eval_timesteps[:, :self.context_length],
                                                eval_task[:, :self.context_length],
                                                actions = None)

        action_dict, pred_actions = self._get_latest_action(action_dict, pred_actions, index=0)  
        eval_actions[:, 0] = pred_actions
        
        with torch.no_grad():
            pred_rtg = self.model(eval_rtg[:, self.context_length],
                                eval_states[:, :self.context_length],
                                eval_timesteps[:, :self.context_length],
                                eval_task[:, :self.context_length],
                                eval_actions[:, self.context_length],
                                eval_rtg = True)
        
        
        pred_rtg = self._get_latest_rtg(pred_rtg, index = 1)
        
        return (eval_states, eval_actions, eval_rtg, eval_rtg, eval_timesteps, eval_task), (states, pred_rtg, pred_actions, action_dict)

        
        
    
    def _generate(self, eval_loader):
            #(Batch_size, 1, 3*128*128), (Batch_size, 1, 1), (Batch_size, 1, 1)
            self.model.eval()
            psnr_increment = 0
            total_reward = 0
            times = []
                
            for index, data in enumerate(eval_loader):
                policy_inputs, mat = data
                
                model_inputs, env_inputs = self.get_initial_policy_setup(policy_inputs, mat)
                eval_states, eval_actions, eval_rtg, eval_rtg, eval_timesteps, eval_task = model_inputs
                states, pred_rtg, _, action_dict = env_inputs
                
                old_reward = self.env.compute_reward(states['x'].real.squeeze(dim = 0), states['gt'])
                
                reward, time, _ = self.run_greedy(states,
                                               pred_rtg,
                                               1,
                                               action_dict,
                                               eval_states, 
                                               eval_actions, 
                                               eval_rtg, 
                                               eval_timesteps, 
                                               eval_task)
                
                
                times.append(time)
                total_reward += reward
                incremental = reward - old_reward
                psnr_increment += incremental
                        
                if (index + 1) % 7 == 0:
                    avg_reward = total_reward/7
                    increment_avg = psnr_increment/7
                    print('Average iter, ', np.mean(times))
                    print('Average reward, ', avg_reward)
                    print('Average ssim: ', average_ssim)
                    print('PSNR increment ', increment_avg)
                    
    
    @torch.no_grad()
    def predict_action_and_rtg(self, eval_states, eval_actions, eval_rtg, eval_timesteps, eval_task, time):
        
        self.model.eval()
        if time < self.context_length:
            pred_actions, action_dict = self.model(eval_rtg[:, :self.context_length], 
                                                    eval_states[:, :self.context_length], 
                                                    eval_timesteps[:, :self.context_length],
                                                    eval_task[:, :self.context_length],
                                                    eval_actions[:, :self.context_length],
                                                    eval_actions = True)
            action_dict, pred_actions = self._get_latest_action(action_dict, pred_actions, index = time)
            eval_actions[:, time] = pred_actions
            pred_rtg = self.model(eval_rtg[:, :self.context_length], 
                                eval_states[:, :self.context_length], 
                                eval_timesteps[:, :self.context_length],
                                eval_task[:, :self.context_length],
                                eval_actions[:, :self.context_length],
                                eval_rtg = True)
            
            pred_rtg = self._get_latest_rtg(pred_rtg, index = time + 1)
                
        else:
            pred_actions, action_dict = self.model(eval_rtg[:,time-self.context_length:time], 
                                                eval_states[:, time-self.context_length:time], 
                                                eval_timesteps[:, time-self.context_length:time],
                                                eval_task[:, time-self.context_length:time],
                                                eval_actions[:, time-self.context_length:time],
                                                eval_actions = True)
            action_dict, pred_actions = self._get_latest_action(action_dict, pred_actions, index = time)
            eval_actions[:, time] = pred_actions
            pred_rtg = self.model(eval_rtg[:,time-self.context_length:time], 
                                eval_states[:, time-self.context_length:time], 
                                eval_timesteps[:, time-self.context_length:time],
                                eval_task[:, time-self.context_length:time],
                                eval_actions[:, time-self.context_length:time],
                                eval_rtg = True)
            
            pred_rtg = self._get_latest_rtg(pred_rtg, index = time + 1)
            
        return pred_actions, action_dict, pred_rtg
                            
                            
    def run_greedy(self,
                   states,
                   pred_rtg,
                   start_time,
                   action_dict,
                   eval_states, 
                   eval_actions, 
                   eval_rtg, 
                   eval_timesteps,
                   eval_task,
                   no_ref = False):
        
        
        for time in range(start_time, self.max_timesteps+1):
            states, done = self.env.step(states, action_dict)
            policy_ob = self.env.get_policy_ob(states)
            
            if (time == self.max_timesteps) or done:
                x = states['x'].reshape(1, 128, 128)
                gt = states['gt']
                if no_ref:
                    reward = self.env.run_no_ref_reward(states)
                else:
                    reward = self.env.compute_reward(x, gt)
                    #temp = x.numpy().reshape(128, 128)* 255
                    #temp = Image.fromarray(temp).convert('RGB')
                    #temp.save('bust_5.png')

                return reward, time, x


            eval_states[:, time] = policy_ob
            eval_rtg[:, time] = pred_rtg
            
            _, action_dict, pred_rtg = self.predict_action_and_rtg(eval_states, eval_actions, eval_rtg, eval_timesteps, eval_task, time)
                    
    def run(self, eval_paths):

        for evalset in eval_paths:
            vanilla_eval_dataset = EvaluationDataset(block_size = self.context_length//3, data_dir=evalset, action_dim= 3, rtg_target = float(self.rtg_target))
            eval_loader = DataLoader(dataset = vanilla_eval_dataset, batch_size=1) 
            increment_reward = self._generate(eval_loader)
            

        

        
            

        
                    
        
        