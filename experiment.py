import os
import torch
from torch import optim
from models import VanillaVAE
from models.types_ import *
import pytorch_lightning as pl
import torchvision.utils as vutils
import numpy as np
from torch.nn import functional as F
import pandas as pd

class VAEXperiment(pl.LightningModule):
    def __init__(self, vae_model: VanillaVAE, params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.validation_outputs = []  
        
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, image: Tensor, **kwargs) -> Tensor:
        return self.model(image, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        img, z_true, z_err = batch
        self.curr_device = img.device
        
        results = self.model(img)

        train_loss = self.model.loss_function(
            *results,
            M_N=self.params['kld_weight'],
            redshift=z_true,
            z_err=z_err, 
            reg_weight=self.params['reg_weight'],
            recon_weight=self.params['recon_weight']
        )

        # 日志记录
        log_dict = {f"train_{key}": val.item() for key, val in train_loss.items() 
                   if key not in ['Pred_Z', 'True_Z', 'Z_ERR']}
        self.log_dict(log_dict, sync_dist=True)
        return {'loss': train_loss['loss']}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        img, z_true, z_err = batch
        self.curr_device = img.device

        results = self.model(img)

        val_loss = self.model.loss_function(
            *results,
            M_N=self.params['kld_weight'],
            redshift=z_true,
            z_err=z_err,
            reg_weight=self.params['reg_weight'],
            recon_weight=self.params['recon_weight']
        )
        
        # 计算红移预测的MAE
        z_pred = val_loss['Pred_Z']
        mae = F.l1_loss(z_pred.squeeze(), z_true)
        
        # 日志记录
        log_dict = {f"val_{key}": val.item() for key, val in val_loss.items() 
                   if key not in ['Pred_Z', 'True_Z', 'Z_ERR']}
        log_dict['val_MAE'] = mae.item()
        self.log_dict(log_dict, sync_dist=True)
        
        # 返回结果用于后续分析（包括delta_z用于计算sigma_NMAD）
        delta_z = (z_pred.squeeze() - z_true).detach().cpu().numpy()
        return {
            'val_loss': val_loss['loss'].item(),
            'pred_z': z_pred.detach().cpu().numpy(),
            'true_z': z_true.detach().cpu().numpy(),
            'z_err': z_err.detach().cpu().numpy(),
            'delta_z': delta_z  # 添加delta_z用于计算sigma_NMAD
        }
    
    def on_validation_epoch_start(self):
        # 清空上一轮的验证结果
        self.validation_outputs = []

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # 收集每个batch的验证结果
        self.validation_outputs.append(outputs)

    def on_validation_epoch_end(self):
        # 检查是否有验证输出
        if not self.validation_outputs:
            return
            
        # 收集所有batch的验证结果
        all_pred_z = np.concatenate([out['pred_z'] for out in self.validation_outputs])
        all_true_z = np.concatenate([out['true_z'] for out in self.validation_outputs])
        all_z_err = np.concatenate([out['z_err'] for out in self.validation_outputs])
        all_delta_z = np.concatenate([out['delta_z'] for out in self.validation_outputs])
        
        # 计算σNMAD
        median_delta_z = np.median(all_delta_z)
        normalized_errors = (all_delta_z - median_delta_z) / (1 + all_true_z)
        sigma_NMAD = 1.4826 * np.median(np.abs(normalized_errors))
        
        # 计算MAE
        mae = np.mean(np.abs(all_delta_z))
        
        # 记录到TensorBoard
        self.log('val_sigma_NMAD', sigma_NMAD, sync_dist=True)
        self.log('val_MAE', mae, sync_dist=True)  # 使用整个验证集计算的MAE
        
        # 保存所有红移预测结果到CSV - 整个验证集
        epoch = self.current_epoch
        redshift_dir = os.path.join(self.logger.log_dir, "Redshift_Predictions")
        os.makedirs(redshift_dir, exist_ok=True)
        csv_path = os.path.join(redshift_dir, f"epoch_{epoch}.csv")
        
        df = pd.DataFrame({
            'true_z': all_true_z,
            'pred_z': all_pred_z.squeeze(),
            'z_err': all_z_err,
            'delta_z': all_delta_z,
            'error': np.abs(all_delta_z),
            'sigma_NMAD': sigma_NMAD  # 添加sigma_NMAD值
        })
        df.to_csv(csv_path, index=False)
        print(f"Saved ALL validation redshift predictions ({len(all_true_z)} samples) to {csv_path}")
        print(f"Validation σNMAD: {sigma_NMAD:.4f}, MAE: {mae:.4f}")
        
        # 采样并保存图像 - 只保存一个批次
        if self.trainer.is_global_zero:  
            self.sample_images()
            
        # 在返回值中添加sigma_NMAD
        return {'val_sigma_NMAD': sigma_NMAD}

    def sample_images(self):
        # 使用验证数据加载器
        val_loader = self.trainer.datamodule.val_dataloader()
        
        # 获取第一个批次的数据
        batch = next(iter(val_loader))
        test_input, test_z, test_z_err = batch
        test_input = test_input.to(self.curr_device)

        recons, residual, z_pred = self.model.generate(test_input)

        # 创建保存路径
        compar_dir = os.path.join(self.logger.log_dir, "Comparison", f"Epoch_{self.current_epoch}")
        os.makedirs(compar_dir, exist_ok=True)


        for c in range(test_input.shape[1]):  # 遍历每个通道
            compar = torch.cat([
                test_input[:, c, :, :].unsqueeze(1),  # 输入图像的当前通道
                recons[:, c, :, :].unsqueeze(1),  # 重建图像的当前通道
                residual[:, c, :, :].unsqueeze(1)  # 残差图像的当前通道
            ], dim=0)  

            compar_path = os.path.join(compar_dir, f"channel_{c}.png")
            vutils.save_image(compar, compar_path, normalize=True, nrow=test_input.shape[0])

        print(f"Saved channel-wise comparison images in {compar_dir}")
        
        # 保存这个批次的红移预测结果
        z_results = []
        for i in range(len(test_z)):
            z_results.append([
                test_z[i].item(), 
                test_z_err[i].item(),  # 保存红移误差
                z_pred[i].item(),
                abs(test_z[i].item() - z_pred[i].item())
            ])
        
        z_df = pd.DataFrame(z_results, columns=['True_Z', 'Z_ERR', 'Pred_Z', 'Error'])
        z_df.to_csv(os.path.join(compar_dir, f"batch_redshift_predictions_epoch_{self.current_epoch}.csv"), index=False)
        
        print(f"Saved comparison images and batch redshift predictions in {compar_dir}")

    def configure_optimizers(self):
        optims = []
        scheds = []
        
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        return optims, scheds
