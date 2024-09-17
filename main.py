import os

import torch
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import CSVLogger

from datamodule import SRDataModule
from lightningmodule import ModelModule
from utils import test_transforms, train_transforms
import argparse
seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def return_call_back():
    return [EarlyStopping('val_loss', patience=5,mode='min'),
            LearningRateMonitor(logging_interval='epoch',log_momentum=True),
            ModelCheckpoint(
                save_top_k=10,
                monitor="val_loss",
                mode="min",
                dirpath="model_ckp",
                filename="SR-{epoch:02d}-{val_loss:.2f}"),
            ]


def create_dataModule(config):
    return SRDataModule(
        train_transforms=train_transforms,
        test_transforms =test_transforms,
        batch_size      = config['data']['batch_size'],
        num_workers     = config['data']['num_workers'],
        drop_last       = config['data']['drop_last'],
        pin_memory=torch.cuda.is_available()
    )
    
def create_modelModule(config):
    return ModelModule(
        window_size= config['model']['window_size'],
        up_scale   = config['model']['scale_factor'],
        img_size   = config['model']['img_size'],
        img_range  = config['model']['img_range'],
        depths     = config['model']['depths'],
        embed_dim  = config['model']['embed_dim'],
        num_heads  = config['model']['num_heads'],
        mlp_ratio  = config['model']['mlp_ratio'],
        upsampler  = config['model']['upsampler'],
        lr         = config['model']['lr']
    )


def main(args):
    config = load_config(args.config)
    
    model = create_modelModule(config)
    datamodule = create_dataModule(config)
    trainer = Trainer(
        deterministic=True, 
        accelerator="auto", 
        callbacks=return_call_back(),
        check_val_every_n_epoch=1,
        fast_dev_run=True,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        logger=CSVLogger(save_dir="logs/"),
        max_epochs=100,
        precision='16',
        enable_progress_bar=True,
        val_check_interval=1.0, # 0.25
        enable_model_summary=True
    )
    trainer.fit(model=model,datamodule=datamodule)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--config','-c' ,help='The path to configuration file')
    args = parser.parse_args()
    main(args)