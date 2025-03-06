from modules_convert import BaseModule
import torch

base_module = BaseModule(obs_dim_dict={"obs1": 64}, module_config_dict={
    "input_dim": ["obs1"],
    "output_dim": [12],
    "layer_config": {
        "type": "MLP",
        "hidden_dims": [32],
        "activation": "ELU"
    }
})
print(base_module)
# 测试转换
scripted_module = torch.jit.script(base_module)
scripted_module.save("base_module_scripted.pt")
print("成功导出为 TorchScript！")
