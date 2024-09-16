from lightning import Trainer
from lightningmodule import ModelModule
from torch.utils.data import DataLoader
from srdataset import YouKuSrDataset
from main import load_config
import argparse
import torch
def load_model(config=None, ckpt:str = None):
    if ckpt:
        return ModelModule.load_from_checkpoint(checkpoint_path=ckpt)
    else:
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
def eval(model,args):
    dataset = YouKuSrDataset()
    dl = DataLoader(dataset=dataset,batch_size=args.b,num_workers=args.num_workers)
    trainer  = Trainer(devices='auto')
    result = trainer.test(model,dl)
    return result
    
def inference(model,input):
    with torch.no_grad():
        prediction = model(input)
    return prediction
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--config','-c' ,help='The path to configuration file')
    parser.add_argument('--mode','-m',choices=['test','inference'])
    parser.add_argument('-bs',default=2,type=int)
    parser.add_argument('--num_workers',default=0,type=int)
    args = parser.parse_args()
    
    config = load_config(args.config)
    model = load_model(config=config)
    dummy_img = torch.randn((1,3,270,480))
    
    if args.mode == 'test':
        result = eval(model,args)
    else:
        result = inference(model,dummy_img)
        print(result.shape)
    