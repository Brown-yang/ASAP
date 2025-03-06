import torch
import torch.nn as nn

###########该类是将modules里面的modules里面的BaseModule全部改写为torch.Tensor类型，支持转换为TorchScript类型
class BaseModule(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict):
        super(BaseModule, self).__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict

        # 用 register_buffer 代替普通变量，确保 TorchScript 兼容
        self.register_buffer("input_dim", torch.tensor(0, dtype=torch.int))
        self.register_buffer("output_dim", torch.tensor(0, dtype=torch.int))

        self._calculate_input_dim()
        self._calculate_output_dim()

        # 初始化 module，避免 TorchScript 解析失败
        self.module = nn.Identity()
        self._build_network_layer(self.module_config_dict["layer_config"])

    def _calculate_input_dim(self):
        input_dim = 0
        for each_input in self.module_config_dict['input_dim']:
            if each_input in self.obs_dim_dict:
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                input_dim += each_input
            else:
                raise ValueError(f"Unknown input type: {each_input}")

        self.input_dim = torch.tensor(input_dim, dtype=torch.int)

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict['output_dim']:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                raise ValueError(f"Unknown output type: {each_output}")

        self.output_dim = torch.tensor(output_dim, dtype=torch.int)

    def _build_network_layer(self, layer_config):
        if layer_config['type'] == 'MLP':
            self._build_mlp_layer(layer_config)
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_config['type']}")

    def _build_mlp_layer(self, layer_config):
        layers = []
        hidden_dims = layer_config['hidden_dims']
        activation = nn.ELU()  # 直接使用固定激活函数，避免 getattr() 解析问题

        layers.append(nn.Linear(self.input_dim.item(), hidden_dims[0]))
        layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], self.output_dim.item()))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)

        self.module = nn.Sequential(*layers)

    def forward(self, input):
        return self.module(input)
