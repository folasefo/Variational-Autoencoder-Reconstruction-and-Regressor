import torch
from models import BaseVAE
from einops import rearrange
from typing import List, Optional, Sequence, Union, Any, Callable
import math
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from .types_ import *

class VanillaVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 regressor_dims: List = [256, 128],
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # 编码器部分
        self.encoder = self.build_encoder(in_channels, hidden_dims)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # 解码器部分
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        decoder_hidden_dims = hidden_dims[::-1]
        self.decoder = self.build_decoder(decoder_hidden_dims)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(decoder_hidden_dims[-1], decoder_hidden_dims[-1],
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(decoder_hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_hidden_dims[-1], out_channels=in_channels,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 回归头
        regressor_layers = []
        input_dim = latent_dim
        for dim in regressor_dims:
            regressor_layers.append(nn.Linear(input_dim, dim))
            regressor_layers.append(nn.LeakyReLU(negative_slope=0.1))
            input_dim = dim
        regressor_layers.append(nn.Linear(input_dim, 1))
        self.regressor = nn.Sequential(*regressor_layers)

    def loss_function(self, *args, **kwargs) -> dict:
        """
        无红移加权版本：完全移除所有红移相关的权重计算
        使用均匀权重和标准的Huber损失
        """
        recons = args[0]
        input  = args[1]
        mu     = args[2]
        log_var= args[3]
        z_pred = args[4]

        kld_weight  = kwargs['M_N']
        z_true      = kwargs['redshift']
        reg_weight  = kwargs['reg_weight']
        recon_weight= kwargs['recon_weight']

        device = z_true.device

        # Reconstruction loss
        recon_loss = F.mse_loss(recons, input)

        # KL divergence
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # 无红移加权：使用均匀权重
        weights = torch.ones_like(z_true, device=device)

        # Huber loss (使用标准delta值)
        z_pred_sq = z_pred.squeeze(-1)
        e = (z_pred_sq - z_true) / (1.0 + z_true)
        huber_loss = F.huber_loss(e, torch.zeros_like(e), delta=0.1, reduction='none')  # 使用标准delta
        regression_loss = (huber_loss * weights).mean()

        # Total loss
        total_loss = (
            recon_weight * recon_loss +
            reg_weight * regression_loss +
            kld_weight * kld_loss
        )

        loss_dict = {
            'loss': total_loss,
            'Reconstruction_Loss': recon_loss.detach(),
            'KLD': kld_loss.detach(),
            'Regression_Loss': regression_loss.detach(),
            'Pred_Z': z_pred.detach(),
            'True_Z': z_true.detach(),
            'Weight_Mean': weights.mean().detach(),
        }

        return loss_dict

    def build_encoder(self, in_channels, hidden_dims):
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.Tanh()
                )
            )
            in_channels = h_dim
        return nn.Sequential(*modules)

    def build_decoder(self, hidden_dims):
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        return nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        z_pred = self.regressor(z)
        return [recons, x, mu, log_var, z_pred]
    
    def load_pretrained_weights(self, checkpoint_path, freeze_encoder=True, freeze_decoder=False):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                pretrained_dict[k[6:]] = v
        self.load_state_dict(pretrained_dict, strict=False)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("✅ 编码器参数已冻结")
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False
            print("✅ 解码器参数已冻结")
        print(f"🔁 预训练权重已加载: {checkpoint_path}")

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        z_pred = self.regressor(z)
        residual = x - recons
        return recons, residual, z_pred