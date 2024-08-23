import torch
import torch.distributions as dist

class Node:
    max_timesteps = 30
    context_length = 6
    
    def __init__(self, rtg, state, time, prob, parent, edge, action_dict, index, policy_state, task) -> None:
        self._parent = parent
        self._children = []
        self.reward = 0
        self.prob = prob
        self.s_visits = 0
        self.time = time
        self.state = state['x'].real.reshape(1, -1)
        self.p_ucb = 0
        self.edge = edge
        self.env_state = state
        self.action_dict = action_dict
        self.index = index
        self.policy_rtg = rtg
        self.policy_state = policy_state
        self.task = task
        
    def __repr__(self) -> str:
        return f"Node(time = {self.time}, edge = {self.edge})_{self.index}"
    
    def set_model_action(self, action: torch.Tensor) -> None:
        self.action = action
        
    def set_action_dict(self, action_dict: torch.Tensor) -> None:
        self.action_dict = action_dict
        
    def backprop(self, reward):
        if reward > self.reward:
            self.reward = reward
            if self._parent != None:
                self._parent.backprop(reward)
            
    def build_eval(self, eval_state,eval_rtg):
        """
        Recursively combine eval state, action, and rtg.
        """
        if self.time < 1:
            eval_state[:, 0] = self.policy_state['x'].real.reshape(1, -1)
            eval_rtg[:, 0] = self.policy_rtg
            return eval_state, eval_rtg
        else:
            eval_state[:, self.time] = self.policy_state['x'].real.reshape(1, -1)
            eval_rtg[:, self.time] = self.policy_rtg
            return self._parent.build_eval(eval_state, eval_rtg)
        
    def build_action(self, eval_actions):
        if self.time < 1:
            eval_actions[:, 0] = self.action
            return eval_actions
        else:
            eval_actions[:, self.time] = self.action
            return self._parent.build_action(eval_actions)
        
        
               

def sample_action_dict(action, prob):
    _distribution = dist.Normal(action.item(), prob)
    action = _distribution.sample(torch.Size([5])).abs()
    probs = torch.exp(_distribution.log_prob(action))
    probs, _indices = torch.sort(probs, descending = True)
    action = action[_indices]
    return action, probs



def select_p_ucb(parent_node, child_nodes, c_base = 10, c = 30):
    max_p_ucb = -1000
    s_visits = parent_node.s_visits
    beta = torch.log(torch.Tensor([(s_visits + c_base + 1)/c_base])) + c
    max_node = parent_node
    
    for node in child_nodes:
        p_ucb = (node.reward - parent_node.reward) + node.prob * torch.sqrt(torch.log(torch.Tensor([s_visits])))/(1 + node.s_visits)
        node.p_ucb = p_ucb
        
        if (p_ucb > max_p_ucb):
            max_node = node
            max_p_ucb = p_ucb

    return max_node  
    
    


def prepare_evaluation(curr_node, task):
    task = task.repeat(1, curr_node.max_timesteps)
    timesteps = torch.arange(0, curr_node.max_timesteps).reshape(1, curr_node.max_timesteps, 1).contiguous()
    eval_actions = torch.zeros((1, curr_node.max_timesteps, 3))
    eval_states = torch.zeros((1, curr_node.max_timesteps, 1*128*128))
    eval_rtg = torch.zeros((1, curr_node.max_timesteps, 1))
    return task, timesteps, eval_actions, eval_states, eval_rtg
    


def expand_tree(evaluator, curr_node, task, env, node_list, index_tree):
    task, timesteps, eval_actions, eval_states, eval_rtg = prepare_evaluation(curr_node, task)
    eval_states, eval_rtg = curr_node.build_eval(eval_states, eval_rtg)
    
    if curr_node._parent:
        eval_actions = curr_node._parent.build_action(eval_actions)
        
    
    pred_actions, action_dict, pred_rtg = evaluator.predict_action_and_rtg(eval_states, eval_actions, eval_rtg, timesteps, task, curr_node.time)

    curr_node.set_model_action(pred_actions)    
    sigma_d, probs = sample_action_dict(action_dict['sigma_d'], 0.2)
    #T, _ = sample_action_dict(action_dict['T'], 0.2)
    mu, probs = sample_action_dict(action_dict['mu'], 0.001)
    
    policy_state, _ = env.step(curr_node.env_state, action_dict)
    
    
    child_nodes = []
    for index in range(len(mu)):
        action_dict['sigma_d'] = sigma_d[index]
        #action_dict['T'] = T[index]
        action_dict['mu'] = mu[index]
        states, _ = env.step(curr_node.env_state, action_dict)
        node = Node(rtg = pred_rtg,
                    state = states, 
                    time = curr_node.time + 1, 
                    prob = probs[index], 
                    parent = curr_node, 
                    edge = index, 
                    action_dict = action_dict,
                    index = index_tree,
                    policy_state=policy_state,
                    task = task)
        
        child_nodes.append(node)
        
        node_list.append(node)
        
    curr_node._children = child_nodes
    return curr_node
    
    
def match_cached_program(node, program_dict):
    for key in program_dict.keys():
        if key == repr(node):
            reward = program_dict[key]
            return reward
    return -100


def find_best(parent_node, child_nodes):
    max_reward = -100
    
    for node in child_nodes:
        if node.reward > max_reward:
            max_reward = node.reward
            max_node = node

    return max_node  


def get_best_program(program_dict, state_dict, node_list, time_dict, env):
    """
    1. get node with best reward ->
    2. recursively extract action dict up until root node
    3. depending on how far is got -> run decision transformer with output
    """
    
    max_reward = -1000
    best_key = None
    
    for key, reward in program_dict.items():
        if reward > max_reward:
            max_reward = reward
            best_key = key
    
    
    for node in node_list:
        if repr(node) == best_key:
            break
    
    print(node)
    state = state_dict[repr(node)]  
    time = time_dict[repr(node)] 
    
    print(time)
    while node._parent:
        node = node._parent  
    return env.compute_reward(node.env_state['gt'].reshape(1, 128, 128), state)
    
    ###initial action
    


def run_beam_search(node, evaluator):
    task, timesteps, eval_actions, eval_states, eval_rtg = prepare_evaluation(node, node.task)
    eval_states, eval_rtg = node.build_eval(eval_states, eval_rtg)
    
    if node._parent:
        eval_actions = node._parent.build_action(eval_actions)
        
    _, action_dict, _ = evaluator.predict_action_and_rtg(eval_states, eval_actions, eval_rtg, timesteps, task, node.time)
    reward, time, final_state = evaluator.run_greedy(node.env_state, node.policy_rtg, node.time, action_dict, eval_states, eval_actions, eval_rtg, timesteps, task, True)
    return reward, final_state, time
            
            
           
        
def run_mcts(eval, policy_inputs, mat, task, env, device_type):
    node_list = []
    _, rtg, _, task = policy_inputs
    states = env.reset(mat, device_type)
    states, rtg = states, rtg.to(device_type)
    
    
    root = Node(rtg, states, 0, 1, None, 0, None, 0, states, task)
    program_dict = {}
    state_dict = {}
    time_dict = {}
    
    node_list.append(root)
    
    # TODO -> need some method of storing nodes
    
    curr_node = root 
    curr_node.s_visits += 1
    
    for i in range(30):
        # TODO do we need visits formula as such???
        curr_node = root
        curr_node.s_visits += 1
        
        #selection
        while len(curr_node._children) > 0:
            curr_node = select_p_ucb(curr_node, curr_node._children)
            curr_node.s_visits += 1
        
        #expansion
        curr_node = expand_tree(eval, curr_node, task, env, node_list, i)
        
        #### if current node reward cached use that
        reward = match_cached_program(curr_node, program_dict)
        
        if reward == -100:
            reward, final_state, time = run_beam_search(curr_node, eval)
            curr_node.reward = reward
            program_dict[repr(curr_node)] = reward
            state_dict[repr(curr_node)] = final_state
            time_dict[repr(curr_node)] = time
            
            
        curr_node.backprop(reward)
    reward = get_best_program(program_dict, state_dict, node_list, time_dict, env)
    print('MCTS Reward: ', reward)
    return reward
    
    



    
            
            
            
    
    
        
        
    
    
    
    
    
    
    
    
        
    
    