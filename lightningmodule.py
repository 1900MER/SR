import lightning as L
import torch
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup

from model import SwinIR
from utils import PSNR


class ModelModule(L.LightningModule):
    def __init__(   
                    self,
                    window_size,
                    up_scale,
                    img_size,
                    img_range,
                    depths,
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    upsampler,
                    lr,
                 ):
        super().__init__()
        # self.hparams.update(hparams)
        self.window_size = window_size
        self.model = SwinIR(upscale = up_scale, img_size=img_size,
                     window_size = self.window_size, img_range = img_range, depths = depths,
                     embed_dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, upsampler = upsampler)
        self.lr = lr
        self.save_hyperparameters(self.hparams)

    def forward(self,x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        print('x',x.shape)
        print('y',y.shape)
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        psnr = PSNR(y_hat,y)
        self.log('val_loss', loss)
        self.log('PSNR',psnr)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                         num_warmup_steps=200, 
                                                         num_training_steps=31250),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,   
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler_config
            }
    
    