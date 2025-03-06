from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal
import inspect

#该类整合了ppo_modules和modules两个文件
#########对原先的Actor和Critic进行整合到一个类里面去 这里该没有转为TensorScript系列
class ActorCritic(nn.Module):
    def __init__(self,
                obs_dim_dict_actor,
                module_config_dict_actor,
                obs_dim_dict_critic,
                module_config_dict_critic,
                num_actions,
                init_noise_std=1.0):
        super(ActorCritic, self).__init__()

        #########需要对传入的参数进行预处理
        module_config_dict_actor = self._process_module_config(module_config_dict_actor, num_actions)
        module_config_dict_critic = self._process_module_config(module_config_dict_critic, num_actions)
        print(module_config_dict_actor)
        print(module_config_dict_critic)


        self.obs_dim_dict_actor = obs_dim_dict_actor
        print(self.obs_dim_dict_actor)
        self.module_config_dict_actor = module_config_dict_actor
        print(self.module_config_dict_actor)

        self.obs_dim_dict_critic = obs_dim_dict_critic
        print(self.obs_dim_dict_critic)
        self.module_config_dict_critic = module_config_dict_critic
        print(self.module_config_dict_critic)
        
        print("wy")
        self._calculate_input_dim_actor()
        print(self.input_dim_actor)
        self._calculate_output_dim_actor()


        self._calculate_input_dim_critic()
        self._calculate_output_dim_crtic()
        print(self.input_dim_critic)
        print(self.output_dim_critic)
        

        # Policy  这里就是利用BaseModule去定义神经网络。
        self.define_actor(self.module_config_dict_actor.layer_config)
        self.define_critic(self.module_config_dict_critic.layer_config)
        print(self.actor)
        print(self.critic)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
    
    def _calculate_input_dim_actor(self):
        # calculate input dimension based on the input specifications
        input_dim_actor = 0
        for each_input in self.module_config_dict_actor['input_dim']:
            if each_input in self.obs_dim_dict_actor:
                # atomic observation type
                input_dim_actor += self.obs_dim_dict_actor[each_input]
            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim_actor += each_input
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")
        
        self.input_dim_actor = input_dim_actor
    
    def _calculate_output_dim_actor(self):
        output_dim_actor = 0
        for each_output in self.module_config_dict_actor['output_dim']:
            if isinstance(each_output, (int, float)):
                output_dim_actor += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")
        self.output_dim_actor = output_dim_actor
    
    def _calculate_input_dim_critic(self):
        # calculate input dimension based on the input specifications
        input_dim_critic = 0
        for each_input in self.module_config_dict_critic['input_dim']:
            if each_input in self.obs_dim_dict_critic:
                # atomic observation type
                input_dim_critic += self.obs_dim_dict_critic[each_input]
            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim_critic += each_input
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")
        
        self.input_dim_critic = input_dim_critic
      
    def _calculate_output_dim_crtic(self):
        output_dim_critic = 0
        for each_output in self.module_config_dict_critic['output_dim']:
            if isinstance(each_output, (int, float)):
                output_dim_critic += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")
        self.output_dim_critic = output_dim_critic


    def define_actor(self, layer_config):
        # Policy
        layers = []
        hidden_dims = layer_config['hidden_dims']
        output_dim = self.output_dim_actor
        activation = getattr(nn, layer_config['activation'])()

        layers.append(nn.Linear(self.input_dim_actor, hidden_dims[0]))
        layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)
        self.actor = nn.Sequential(*layers)
    

    def define_critic(self, layer_config):
        # Policy
        layers = []
        hidden_dims = layer_config['hidden_dims']
        output_dim = self.output_dim_critic
        activation = getattr(nn, layer_config['activation'])()

        layers.append(nn.Linear(self.input_dim_critic, hidden_dims[0]))
        layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)
        self.critic = nn.Sequential(*layers)

    
    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
       

    
    

