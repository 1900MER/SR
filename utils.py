import cv2
import time
import torch
import torch.nn.functional as F
import math
import random
import torchvision
class RandomApply(object):
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img
    
    def __repr__(self):
        return f"RandomApply({self.transform.__repr__()}, p={self.p})"

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
    
train_transforms = torchvision.transforms.Compose(
    [
        
        RandomApply(torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), p=0.8),
        RandomApply(AddGaussianNoise(0, 0.05), p=0.5),
        RandomApply(torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), p=0.5),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Lambda(lambda x: x / 255.)
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Lambda(lambda x: x / 255.)
    ]
)

def PSNR(img1,img2,max_value  = 1.):
    # 确保输入是PyTorch张量
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1).float()
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2).float()
    
    # 确保图像在正确的范围内
    img1 = img1.clamp(0, max_value)
    img2 = img2.clamp(0, max_value)
    
    # 计算MSE (Mean Squared Error)
    mse = F.mse_loss(img1, img2)
    
    # 计算PSNR
    psnr = 10 * torch.log10(max_value**2 / mse)
    
    return psnr.item()


gt_l_path = {
    'video':{
        'gt':'data/H_GT',
         'l':'data/L'
    },
    'image':{
        'gt':'video_image/H_GT',
        'l' :'video_image/L'
    }
}

def read_img(file_path:str = None):
    img = cv2.imread(file_path)
    cv2.imshow(file_path,img)
    cv2.waitKey(0)
    return img
    

def timer(func):
    def wrapper(*args, **kw):
        start = time.time()
        print('===== start =====')
        func(*args, **kw)
        end   = time.time()
        print('===== End =====')
        print(f'Total Time: {end-start}')
    return wrapper   

def create_formatted_number(number_length:int = None,number:int = None):
    str_number = str(number)
    return '0'*(number_length-len(str_number))+str_number   
    
def total_image(h_gt_suffix = None,l_suffix = None,video_number:str= None,frames_per_video:str=None, video_ext =None):
    gt_images_dir = 'video_image/H_GT/Youku_'+create_formatted_number(5,video_number) + h_gt_suffix + '/image_'
    l_images_dir  = 'video_image/L/Youku_'   +create_formatted_number(5,video_number) + l_suffix    + '/image_'
    
    return [ (gt_images_dir+create_formatted_number(4,number)+video_ext,l_images_dir+create_formatted_number(4,number)+video_ext) for number in range(1,frames_per_video+1)]


