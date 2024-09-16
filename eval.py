from lightning import Trainer
from lightningmodule import ModelModule
from torch.utils.data import DataLoader
from srdataset import YouKuSrDataset
from main import load_config
import argparse
import torch
from torchvision import transforms
from PIL import Image

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
    
def inference(model,img_path):
    transform = transforms.Compose([
        transforms.Resize((270,480)),
        transforms.ToTensor(),  # 将图像转换为 tensor，并将像素值范围调整到 [0, 1]
    ])
    
    img = Image.open(img_path)
    resized_img = transform(img)
    if resized_img.shape != (1,3,270,480):
        resized_img = resized_img.unsqueeze(0)
    with torch.no_grad():
        prediction = model(resized_img)

    pil = transforms.ToPILImage()(prediction[0]*255)
    pil.show()
    return pil
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--config','-c' ,help='The path to configuration file')
    parser.add_argument('--mode','-m',choices=['test','inference'])
    parser.add_argument('-bs',default=2,type=int)
    parser.add_argument('--num_workers',default=0,type=int)
    parser.add_argument('--img_path')
    args = parser.parse_args()
    
    config = load_config(args.config)
    model = load_model(config=config)
    
    if args.mode == 'test':
        result = eval(model,args)
    else:
        result = inference(model,args.img_path)        
    