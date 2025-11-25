import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning import Callback
from dataset import VAEDataset
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import seed_everything
import torch
import re
import pytorch_lightning as pl
import copy


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='/home/wmee/PyTorch-VAE/configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
        max_epochs = config['trainer_params']['max_epochs']
        early_stop_params = config.get('early_stop_params', {})
        freeze_params = config.get('freeze_params', {})
        
        # 获取预训练模型路径
        pretrain_params = config.get('pretrain_params', {})
        pretrained_path = pretrain_params.get('checkpoint_path', '')
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['model_params']['name'],
)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

# 创建模型
model = vae_models[config['model_params']['name']](**config['model_params'])

# ====== 关键修改：加载预训练冻结模型 ======
if pretrained_path and os.path.exists(pretrained_path):
    print(f"🚀 加载预训练冻结模型: {pretrained_path}")
    
    # 加载状态字典
    state_dict = torch.load(pretrained_path, map_location='cpu')
    
    # 提取模型权重（可能包含'state_dict'键）
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # 调整键名：去掉前缀（例如：'model.'）
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            k = k[6:]  # 去掉 'model.' 前缀
        pretrained_dict[k] = v
    
    # 加载权重到当前模型（严格模式关闭）
    model.load_state_dict(pretrained_dict, strict=False)
    
    # 冻结指定模块
    freeze_encoder = config['freeze_params'].get('freeze_encoder', True)
    freeze_decoder = config['freeze_params'].get('freeze_decoder', True)
    
    if freeze_encoder:
        # 冻结编码器参数
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("✅ 编码器参数已冻结")
    
    if freeze_decoder:
        # 冻结解码器参数
        for param in model.decoder.parameters():
            param.requires_grad = False
        print("✅ 解码器参数已冻结")
    
    # 打印可训练参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 可训练参数: {trainable_params}/{total_params} ({trainable_params/total_params:.2%})")
else:
    print("⚠️ 未找到预训练模型，将从头开始训练")

config['exp_params']['label_csv'] = config['data_params']['label_csv']
config['exp_params']['bin_width'] = config['data_params'].get('bin_width', 0.1)
config['exp_params']['max_z'] = config['data_params'].get('max_z', 0.7)
# 这两个开关/强度也传一下（若 YAML 里没配，会用默认值）
config['exp_params']['use_bin_loss_weight'] = config['exp_params'].get('use_bin_loss_weight', True)
config['exp_params']['bin_weight_alpha']   = config['exp_params'].get('bin_weight_alpha', 0.5)

# 创建实验
experiment = VAEXperiment(model, config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()

# 定义早停回调
early_stop_callback = EarlyStopping(
    monitor=early_stop_params.get('monitor', 'val_loss'),
    min_delta=early_stop_params.get('min_delta', 0.0005),
    patience=early_stop_params.get('patience', 15),
    verbose=early_stop_params.get('verbose', True),
    mode=early_stop_params.get('mode', 'min'),
)

# 创建ModelCheckpoint回调并保留引用
checkpoint_callback = ModelCheckpoint(
    save_top_k=2,
    dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
    monitor="val_loss",
    save_last=True,
    mode="min",
    filename='{epoch}-{val_loss:.4f}'  # 文件名包含epoch和loss值
)

# 为sigma_NMAD创建单独的回调
sigma_nmad_callback = ModelCheckpoint(
    monitor="val_sigma_NMAD",
    dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
    filename="best_sigma_epoch={epoch}-sigma_NMAD={val_sigma_NMAD:.4f}",
    save_top_k=1,
    mode="min",
    save_last=False
)

class FreezeCallback(Callback):
    def __init__(self, freeze_modules=['encoder', 'decoder', 'fc_mu', 'fc_var'], save_frozen_model=True):
        super().__init__()
        self.freeze_modules = freeze_modules
        self.save_frozen_model = save_frozen_model
        self.best_sigma = float('inf')
        self.best_sigma_epoch = -1
        
    def on_validation_end(self, trainer, pl_module):
        """在每个验证周期结束时检查sigma_NMAD"""
        current_sigma = trainer.callback_metrics.get('val_sigma_NMAD', None)
        
        if current_sigma is not None and current_sigma < self.best_sigma:
            self.best_sigma = current_sigma
            self.best_sigma_epoch = trainer.current_epoch
            print(f"🌟 新的最佳sigma_NMAD: {self.best_sigma:.4f} (epoch {self.best_sigma_epoch})")
            
            # 保存最佳sigma_NMAD模型
            ckpt_path = os.path.join(
                trainer.logger.log_dir, 
                "checkpoints", 
                f"best_sigma_epoch={self.best_sigma_epoch}-sigma_NMAD={self.best_sigma:.4f}.ckpt"
            )
            trainer.save_checkpoint(ckpt_path)
            print(f"💾 最佳sigma_NMAD模型已保存至: {ckpt_path}")
    
    def on_train_end(self, trainer, pl_module):
        """训练结束后冻结最佳sigma_NMAD模型的参数"""
        if self.best_sigma_epoch == -1:
            print("⚠️ 未检测到最佳sigma_NMAD，跳过冻结操作")
            return
            
        # 加载最佳sigma_NMAD模型
        ckpt_path = os.path.join(
            trainer.logger.log_dir, 
            "checkpoints", 
            f"best_sigma_epoch={self.best_sigma_epoch}-sigma_NMAD={self.best_sigma:.4f}.ckpt"
        )
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ 找不到最佳sigma_NMAD模型: {ckpt_path}")
            return
            
        print(f"\n🔥 在最佳sigma_NMAD模型上冻结参数 (epoch {self.best_sigma_epoch}, σ_NMAD={self.best_sigma:.4f})")
        
        # 加载模型状态
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # ====== 关键修复：处理状态字典键名 ======
        # 创建新的状态字典，移除多余的键名
        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除 'model.' 前缀
            if k.startswith('model.'):
                k = k[6:]  # 去掉 'model.' 前缀
            # 忽略 num_batches_tracked 参数
            if 'num_batches_tracked' in k:
                continue
            new_state_dict[k] = v
        
        # 创建新模型实例
        model = pl_module.model
        
        # 加载权重（使用 strict=False 忽略不匹配的键）
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ 模型权重加载成功")
        
        # 冻结指定模块
        for module_name in self.freeze_modules:
            module = getattr(model, module_name, None)
            if module is None:
                print(f"⚠️ 找不到模块 '{module_name}'，跳过冻结")
                continue
                
            for param in module.parameters():
                param.requires_grad = False
            print(f"✅ 模块 '{module_name}' 已冻结")
        
        # 保存冻结状态模型
        if self.save_frozen_model:
            freeze_path = os.path.join(
                trainer.logger.log_dir, 
                "checkpoints", 
                f"frozen_sigma_epoch={self.best_sigma_epoch}-sigma_NMAD={self.best_sigma:.4f}.ckpt"
            )
            
            # 保存冻结模型
            torch.save({
                'state_dict': model.state_dict(),
                'hyper_parameters': {
                    'best_sigma': self.best_sigma,
                    'best_epoch': self.best_sigma_epoch,
                    'freeze_modules': self.freeze_modules
                }
            }, freeze_path)
            
            print(f"💾 冻结状态模型已保存至: {freeze_path}")

# 从配置获取冻结参数
freeze_modules = freeze_params.get('freeze_modules', ['encoder', 'decoder', 'fc_mu', 'fc_var'])
save_frozen_model = freeze_params.get('save_frozen_model', True)

# 创建冻结回调
freeze_callback = FreezeCallback(
    freeze_modules=freeze_modules,
    save_frozen_model=save_frozen_model
)

# 关键修改：更新DDP策略以支持冻结参数
ddp_strategy = DDPStrategy(
    find_unused_parameters=True  # 允许未使用的参数
)

runner = Trainer(
    logger=tb_logger,
    callbacks=[ 
        LearningRateMonitor(),
        checkpoint_callback,
        sigma_nmad_callback,
        early_stop_callback,
        freeze_callback
    ],
    strategy=ddp_strategy,
    replace_sampler_ddp=False,   # ★ 关键：让我们在 DataModule 里传入的 WeightedRandomSampler 生效
    **config['trainer_params']
)

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")
print(f"冻结模块: {freeze_modules}")
torch.cuda.empty_cache()
runner.fit(experiment, datamodule=data)

if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
    best_model_filename = os.path.basename(checkpoint_callback.best_model_path)

    # 解析 epoch
    epoch = "未知"
    m_epoch = re.search(r'epoch=(\d+)', best_model_filename)
    if m_epoch:
        epoch = int(m_epoch.group(1))
    else:
        # 对形如 "51-0.0011.ckpt" 的文件名，从 stem 抓取
        stem = Path(best_model_filename).stem  # "51-0.0011"
        parts = stem.split('-')
        if parts and parts[0].isdigit():
            epoch = int(parts[0])

    # 解析 val_loss
    best_loss = None
    m_loss = re.search(r'val_loss=([0-9]*\.?[0-9]+)', best_model_filename)
    if m_loss:
        best_loss = float(m_loss.group(1))
    else:
        # 对形如 "51-0.0011.ckpt"：取去后缀后的最后一段
        try:
            best_loss = float(Path(best_model_filename).stem.split('-')[-1])
        except Exception:
            # 最后兜底
            best_loss = checkpoint_callback.best_model_score.item()

    print(f"\n{'='*50}")
    print(f"🏆 最佳验证损失模型信息:")
    print(f"📁 路径: {checkpoint_callback.best_model_path}")
    print(f"🔢 Epoch: {epoch}")
    print(f"📉 最低验证Loss: {best_loss:.6f}")
    print(f"{'='*50}")

    with open(os.path.join(tb_logger.log_dir, "best_results.txt"), "w") as f:
        f.write(f"最佳验证损失模型路径: {checkpoint_callback.best_model_path}\n")
        f.write(f"最佳Epoch: {epoch}\n")
        f.write(f"最低验证Loss: {best_loss:.6f}\n")
        
# 显示最佳sigma_NMAD结果
if freeze_callback.best_sigma_epoch != -1:
    print(f"\n{'='*50}")
    print(f"🏆 最佳sigma_NMAD模型信息:")
    print(f"🔢 Epoch: {freeze_callback.best_sigma_epoch}")
    print(f"📉 最低σ_NMAD: {freeze_callback.best_sigma:.6f}")
    print(f"💾 冻结模型路径: {os.path.join(tb_logger.log_dir, 'checkpoints', f'frozen_sigma_epoch={freeze_callback.best_sigma_epoch}-sigma_NMAD={freeze_callback.best_sigma:.4f}.ckpt')}")
    print(f"{'='*50}")
    
    with open(os.path.join(tb_logger.log_dir, "best_results.txt"), "a") as f:
        f.write(f"\n最佳sigma_NMAD模型信息:\n")
        f.write(f"Epoch: {freeze_callback.best_sigma_epoch}\n")
        f.write(f"σ_NMAD: {freeze_callback.best_sigma:.6f}\n")
        f.write(f"冻结模型路径: {os.path.join(tb_logger.log_dir, 'checkpoints', f'frozen_sigma_epoch={freeze_callback.best_sigma_epoch}-sigma_NMAD={freeze_callback.best_sigma:.4f}.ckpt')}\n")
else:
    print("⚠️ 未找到最佳sigma_NMAD模型信息")

print("训练完成！")