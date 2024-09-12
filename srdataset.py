from typing import Any, Union
import PIL.Image
import cv2
from utils import total_image
from torch.utils.data import Dataset
import random
import PIL
class YouKuSrDataset(Dataset):
    def __init__(self, max_video:int = 50,
                       frames_video:str = 100,
                       image_ext = '.jpg',
                       split:str = None,
                       val_video = None,
                       transforms = None):
        super().__init__()
        self.max_video = max_video
        self.frames_video = frames_video
        self.image_ext = image_ext
        self.h_gt_suffix = '_h_GT'
        self.l = '_l'
        self.data = []
        
        if split == 'train':
            start = 0
            end = self.max_video - val_video
        elif split == 'val':
            start = self.max_video - val_video
            end = self.max_video
        else:
            raise ValueError('Split must be either train or val')
        
        for i in range(start,end):
            self.data.extend(total_image(self.h_gt_suffix,
                                         self.l,
                                         i,
                                         self.frames_video,
                                         self.image_ext)
                             )
        
        random.shuffle(self.data)
        self.transforms = transforms
        
    def __getitem__(self,index):
        gt_path, l_path = self.data[index]
        gt_image = PIL.Image.open(gt_path)
        l_image  = PIL.Image.open(l_path)
        if self.transforms:
            gt_image = self.transforms(gt_image)
            l_image  = self.transforms(l_image)
        
        # return (torch.from_numpy(gt_image).permute(2,0,1), torch.from_numpy(l_image).permute(2,0,1))
        return l_image,gt_image
        
        
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    ds = YouKuSrDataset()
    print(ds[10][0].shape)
        
        
