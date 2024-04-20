# Author: Vibhakar Mohta vmohta@cs.cmu.edu
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import itertools
import pickle
import math
import numpy as np
from collections import OrderedDict
from typing import Union, Callable
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append("/home/ros_ws/")

#@markdown ### **Network**
#@markdown
#@markdown Defines a 2D UNet architecture `ConditionalUnet1D`
#@markdown as the noies prediction network
#@markdown
#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv2dBlock(nn.Module):
    '''
        Conv2d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv2dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv2dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation which predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels)
        )

        # If the number of input channels is not equal to output channels,
        # a 1x1 convolution is used to match the dimensions
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [B, C, H, W]
            cond : [B, cond_dim]

            returns:
            out : [B, out_channels, H, W]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        
        # Reshape the embed to [B, 2 * out_channels, 1, 1]
        # to apply it across the spatial dimensions H and W
        embed = embed.view(-1, 2, self.out_channels, 1, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]

        # Apply FiLM conditioning
        out = scale * out + bias

        out = self.blocks[1](out)
        
        # Apply the residual connection
        residual = self.residual_conv(x)
        out += residual

        return out

class ConditionalUnet2D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[8,16,32],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        channels, height, width = input_dim
        self.channels = channels

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        all_dims = [channels] + list(down_dims)
        start_dim = down_dims[0]

        # Create down_modules
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])
        
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock2D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Downsample2d(dim_out) if not is_last else nn.Identity()
            ]))

        # Create up_modules
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(dim_out*2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock2D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Upsample2d(dim_in) if not is_last else nn.Identity()
            ]))

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(start_dim, start_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.Conv2d(start_dim, channels, kernel_size=1)
        )

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, sample: torch.Tensor, timestep, global_cond=None):
        """
        sample: Input tensor of shape (B, C, H, W)
        timestep: Single int or tensor indicating the diffusion step
        global_cond: Optional tensor of shape (B, global_cond_dim)

        Returns:
            Tensor of shape (B, C, H, W) after processing through Conditional UNet2D
        """
        # Process timestep input to ensure it's a tensor and properly broadcasted
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).to(sample.device)
        timestep = timestep.expand(sample.size(0))  # ensure it has batch size dimension

        # Generate global features combining timestep embeddings and any additional global conditioning
        global_feature = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)
        
        # global_feature shape (B, global_cond_dim = 768)

        # Encoder: progressively downsample
        h = []
        x = sample
        print("INPUT SHAPE: ", x.shape)
        print("Downsample")
        for resnet, resnet2, downsample in self.down_modules:
            print("Input shape: ", x.shape)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        print("Shape after downsample: ", x.shape)
        # Bottleneck: process at the lowest resolution
        print("Mid modules")
        for mid_module in self.mid_modules:
            print("Input shape: ", x.shape)
            x = mid_module(x, global_feature)

        # Decoder: progressively upsample and concatenate skip connections
        print("Upsample")
        for resnet, resnet2, upsample in self.up_modules:
            print("Input shape: ", x.shape)
            x = torch.cat((x, h.pop()), dim=1)  # concatenate along the channel dimension
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        print("Shape before final conv: ", x.shape)
        # Final convolution layer to map back to original dimensions
        x = self.final_conv(x)
        print("Shape after final conv: ", x.shape)
        return x
    


#@markdown ### **Vision Encoder**
#@markdown
#@markdown Defines helper functions:
#@markdown - `get_resnet` to initialize standard ResNet vision encoder
#@markdown - `replace_bn_with_gn` to replace all BatchNorm layers with GroupNorm

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def get_vision_encoder(
        name:str,
        weights=None,
        **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet(name)

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    return vision_encoder

if __name__=="__main__":
    # Get ConditionalUnet2D
    input_dim = (3, 64, 64)
    global_cond_dim = 512
    model = ConditionalUnet2D(input_dim, global_cond_dim)
    
    # Get vision encoder
    vision_encoder = get_vision_encoder("resnet18")
    
    # dummy input
    x = torch.randn(4, 3, 64, 64)
    timestep = 0
    global_cond = vision_encoder(x)
    
    # forward pass
    rand_noise = torch.randn(4, 3, 64, 64)
    out = model(rand_noise, timestep, global_cond)
    
    print(out.shape)