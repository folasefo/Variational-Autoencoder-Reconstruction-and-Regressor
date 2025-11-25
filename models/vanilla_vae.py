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
                 z_min: float = 0.0,
                 z_max: float = 0.7,
                 z0: float = 0.3161643944329,          # 开始增强高z的阈值
                 gamma_low: float = 2.2965550283044145,   # 低z段指数
                 gamma_high: float = 3.690762747549663,  # 高z段指数（越大越强调高z）
                 gamma_k: float = 32.934810657497884,    # 过渡锐度
                 beta_err: float = 0.38191863876162563,    # 误差权重幂次 (1/z_err)^beta
                 weight_clamp_min: float = 0.6607841682826632,
                 weight_clamp_max: float = 10.732496869073254,
                 huber_delta_e: float = 0.013428681496104579,  # Huber作用于归一化残差的δ
                 # lambda_trend: float = 0.0,   
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # 保存超参
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.z0 = float(z0)
        self.gamma_low = float(gamma_low)
        self.gamma_high = float(gamma_high)
        self.gamma_k = float(gamma_k)
        self.beta_err = float(beta_err)
        self.weight_clamp_min = float(weight_clamp_min)
        self.weight_clamp_max = float(weight_clamp_max)
        self.huber_delta_e = float(huber_delta_e)

        # 编码器部分
        self.encoder = self.build_encoder(in_channels, hidden_dims)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # 解码器部分
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        decoder_hidden_dims = hidden_dims[::-1]  # [512, 256, 128, 64, 32]
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
        z_pred = self.regressor(z)  # 红移预测
        return [recons, x, mu, log_var, z_pred]
    
    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input  = args[1]
        mu     = args[2]
        log_var= args[3]
        z_pred = args[4]  # (B,1)

        kld_weight  = kwargs['M_N']
        z_true      = kwargs['redshift']              # (B,)
        z_err       = kwargs.get('z_err', None)       # (B,)
        reg_weight  = kwargs['reg_weight']
        recon_weight= kwargs['recon_weight']

        device = z_true.device
        eps_small = torch.tensor(1e-6, device=device)

        # --- Reconstruction ---
        recon_loss = F.mse_loss(recons, input)

        # --- KL ---
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # --- Measurement-uncertainty base weight: (1 / z_err)^beta (温和) ---
        if z_err is not None:
            min_z_err = torch.tensor(1e-3, device=device)
            base_weights = torch.pow(1.0 / torch.max(z_err, min_z_err), self.beta_err)
            base_weights = base_weights / (base_weights.mean() + eps_small)
        else:
            base_weights = torch.ones_like(z_true, device=device)

        # --- z-dependent weight with smooth steepening after z0 ---
        t = ((z_true - self.z_min) / max(self.z_max - self.z_min, 1e-6))
        t = torch.clamp(t, 0.0, 1.0) + 1e-6
        t0 = (self.z0 - self.z_min) / max(self.z_max - self.z_min, 1e-6)

        gamma_t = self.gamma_low + (self.gamma_high - self.gamma_low) * torch.sigmoid(
            torch.tensor(self.gamma_k, device=device) * (t - t0)
        )

        # 高红移样本加权
        high_z_weight = torch.sigmoid(z_true * 10)  # 红移值越高，权重越大

        cont_w = torch.pow(t, gamma_t) * high_z_weight

        # --- combine & clamp & renorm(mean=1) ---
        weights = base_weights * cont_w
        weights = torch.clamp(weights, self.weight_clamp_min, self.weight_clamp_max)
        weights = weights / (weights.mean() + eps_small)

        # --- Huber on normalized residual e = (z_pred - z_true)/(1+z_true) ---
        z_pred_sq = z_pred.squeeze(-1)  # (B,)
        e = (z_pred_sq - z_true) / (1.0 + z_true)
        huber_loss = F.huber_loss(e, torch.zeros_like(e), delta=self.huber_delta_e, reduction='none')
        regression_loss = (huber_loss * weights).mean()

        # --- total loss ---
        total_loss = (
            recon_weight * recon_loss
            + reg_weight * regression_loss
            + kld_weight  * kld_loss
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
        if z_err is not None:
            loss_dict['Z_ERR'] = z_err.detach().mean()

        return loss_dict

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        z_pred = self.regressor(z)  # 预测红移
        residual = x - recons
        return recons, residual, z_pred
