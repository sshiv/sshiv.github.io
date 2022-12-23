---
title: "Get all Input & Output Sizes of for Layers of a PyTorch Model"
excerpt_separator: "<!--more-->"
categories:
  - Blog
tags:
  - python
  - pytorch
---

One way to get the input and output sizes for Layers/Modules in a PyTorch model is to register a forward hook using [torch.nn.modules.module.register_module_forward_hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html). 
The hook function gets called every time forward is called on the registered module. Conversely all the modules you need information from need to be explicity registered. The same method could be used to get the activations and other data per layer.

<!--more-->
The following code shows a simple way of wrapping this up to print the input output summary for a pytorch model.

```python

import torch
from torch import nn

class TorchModelSummary:
    """ Class to collect and Pytorch Module information"""
    def __init__(self):
        """ Initialize the maps """
        self.module_summary_map = {}
        self.module_hook_map = {}
    
    def get_forward_hook(self, module_name : str):
        """ Wrap the hook to associate the module name """
        def forward_hook_fn(module : nn.Module, input_data : torch.Tensor, output : torch.Tensor):
            self.module_summary_map[module_name] = {"name": module_name,
                                                    "input_size" : input_data[0].shape,
                                                    "output_size" : output[0].shape}
        return forward_hook_fn
    
    def torch_size_to_str(tensor_size : torch.Size) -> str:
        """ Simple helper function to pretty print torch size """
        return " x ".join(map(str, tensor_size))
    
    def print_summary(self):
        """ Print the summary of the model """
        header = str.format("{0:<30}| {1:<30}| {2:<30}|", "Module Name", "Input Size", "Output Size")
        print(header)
        print("-" * len(header))
        for module_name in self.module_summary_map:
            summary = self.module_summary_map[module_name]
            print(f'{module_name:<30}|'
                  f' {TorchModelSummary.torch_size_to_str(summary["input_size"]):<30}|'
                  f' {TorchModelSummary.torch_size_to_str(summary["output_size"]):<30}|')
         
    def __call__(self, module : nn.Module, x: torch.Tensor):
        """ Register the hooks, call forward, print summary and remove hooks """
        for module_name, module in module.named_modules():
            self.module_hook_map[module_name] = 
              module.register_forward_hook(self.get_forward_hook(module_name))
        _ = module(x)
        
        self.print_summary()
        
        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in module.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} trainable parameters.")
        
        # Remove all the hooks
        for module_name in self.module_hook_map:
            self.module_hook_map[module_name].remove()
        
def PrintTorchModelSummary(model : nn.Module, x : torch.Tensor):
    """ Simple wrapper for calling TorchModelSummary """
    tms = TorchModelSummary()
    tms(model, x)
    
# Sample Usage
from torchvision.models import resnet18
model = resnet18()
data = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
PrintTorchModelSummary(model, data)
```
Sample output
```cmd
Module Name                   | Input Size                    | Output Size                   |
-----------------------------------------------------------------------------------------------
conv1                         | 1 x 3 x 224 x 224             | 64 x 112 x 112                |
bn1                           | 1 x 64 x 112 x 112            | 64 x 112 x 112                |
relu                          | 1 x 64 x 112 x 112            | 64 x 112 x 112                |
maxpool                       | 1 x 64 x 112 x 112            | 64 x 56 x 56                  |
layer1.0.conv1                | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1.0.bn1                  | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1.0.relu                 | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1.0.conv2                | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1.0.bn2                  | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1.0                      | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1.1.conv1                | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1.1.bn1                  | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1.1.relu                 | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1.1.conv2                | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1.1.bn2                  | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1.1                      | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer1                        | 1 x 64 x 56 x 56              | 64 x 56 x 56                  |
layer2.0.conv1                | 1 x 64 x 56 x 56              | 128 x 28 x 28                 |
layer2.0.bn1                  | 1 x 128 x 28 x 28             | 128 x 28 x 28                 |
layer2.0.relu                 | 1 x 128 x 28 x 28             | 128 x 28 x 28                 |
layer2.0.conv2                | 1 x 128 x 28 x 28             | 128 x 28 x 28                 |
layer2.0.bn2                  | 1 x 128 x 28 x 28             | 128 x 28 x 28                 |
layer2.0.downsample.0         | 1 x 64 x 56 x 56              | 128 x 28 x 28                 |
layer2.0.downsample.1         | 1 x 128 x 28 x 28             | 128 x 28 x 28                 |
layer2.0.downsample           | 1 x 64 x 56 x 56              | 128 x 28 x 28                 |
layer2.0                      | 1 x 64 x 56 x 56              | 128 x 28 x 28                 |
layer2.1.conv1                | 1 x 128 x 28 x 28             | 128 x 28 x 28                 |
layer2.1.bn1                  | 1 x 128 x 28 x 28             | 128 x 28 x 28                 |
layer2.1.relu                 | 1 x 128 x 28 x 28             | 128 x 28 x 28                 |
layer2.1.conv2                | 1 x 128 x 28 x 28             | 128 x 28 x 28                 |
layer2.1.bn2                  | 1 x 128 x 28 x 28             | 128 x 28 x 28                 |
layer2.1                      | 1 x 128 x 28 x 28             | 128 x 28 x 28                 |
layer2                        | 1 x 64 x 56 x 56              | 128 x 28 x 28                 |
layer3.0.conv1                | 1 x 128 x 28 x 28             | 256 x 14 x 14                 |
layer3.0.bn1                  | 1 x 256 x 14 x 14             | 256 x 14 x 14                 |
layer3.0.relu                 | 1 x 256 x 14 x 14             | 256 x 14 x 14                 |
layer3.0.conv2                | 1 x 256 x 14 x 14             | 256 x 14 x 14                 |
layer3.0.bn2                  | 1 x 256 x 14 x 14             | 256 x 14 x 14                 |
layer3.0.downsample.0         | 1 x 128 x 28 x 28             | 256 x 14 x 14                 |
layer3.0.downsample.1         | 1 x 256 x 14 x 14             | 256 x 14 x 14                 |
layer3.0.downsample           | 1 x 128 x 28 x 28             | 256 x 14 x 14                 |
layer3.0                      | 1 x 128 x 28 x 28             | 256 x 14 x 14                 |
layer3.1.conv1                | 1 x 256 x 14 x 14             | 256 x 14 x 14                 |
layer3.1.bn1                  | 1 x 256 x 14 x 14             | 256 x 14 x 14                 |
layer3.1.relu                 | 1 x 256 x 14 x 14             | 256 x 14 x 14                 |
layer3.1.conv2                | 1 x 256 x 14 x 14             | 256 x 14 x 14                 |
layer3.1.bn2                  | 1 x 256 x 14 x 14             | 256 x 14 x 14                 |
layer3.1                      | 1 x 256 x 14 x 14             | 256 x 14 x 14                 |
layer3                        | 1 x 128 x 28 x 28             | 256 x 14 x 14                 |
layer4.0.conv1                | 1 x 256 x 14 x 14             | 512 x 7 x 7                   |
layer4.0.bn1                  | 1 x 512 x 7 x 7               | 512 x 7 x 7                   |
layer4.0.relu                 | 1 x 512 x 7 x 7               | 512 x 7 x 7                   |
layer4.0.conv2                | 1 x 512 x 7 x 7               | 512 x 7 x 7                   |
layer4.0.bn2                  | 1 x 512 x 7 x 7               | 512 x 7 x 7                   |
layer4.0.downsample.0         | 1 x 256 x 14 x 14             | 512 x 7 x 7                   |
layer4.0.downsample.1         | 1 x 512 x 7 x 7               | 512 x 7 x 7                   |
layer4.0.downsample           | 1 x 256 x 14 x 14             | 512 x 7 x 7                   |
layer4.0                      | 1 x 256 x 14 x 14             | 512 x 7 x 7                   |
layer4.1.conv1                | 1 x 512 x 7 x 7               | 512 x 7 x 7                   |
layer4.1.bn1                  | 1 x 512 x 7 x 7               | 512 x 7 x 7                   |
layer4.1.relu                 | 1 x 512 x 7 x 7               | 512 x 7 x 7                   |
layer4.1.conv2                | 1 x 512 x 7 x 7               | 512 x 7 x 7                   |
layer4.1.bn2                  | 1 x 512 x 7 x 7               | 512 x 7 x 7                   |
layer4.1                      | 1 x 512 x 7 x 7               | 512 x 7 x 7                   |
layer4                        | 1 x 256 x 14 x 14             | 512 x 7 x 7                   |
avgpool                       | 1 x 512 x 7 x 7               | 512 x 1 x 1                   |
fc                            | 1 x 512                       | 1000                          |
                              | 1 x 3 x 224 x 224             | 1000                          |
513,000 total parameters.
513,000 trainable parameters.

```
