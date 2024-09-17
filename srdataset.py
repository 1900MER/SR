from typing import Any, Union
import PIL.Image
import cv2
from utils import total_image
from torch.utils.data import Dataset
import random
import PIL
import os
class YouKuSrDataset(Dataset):
    def __init__(self,
                       frames_video:str = 100,
                       image_ext = '.jpg',
                       split:str = None,
                       transforms = None,
                       root_dir:str = None
                ):
        super().__init__()
        self.split = split
        self.frames_video = frames_video
        self.image_ext = image_ext
        self.h_gt_suffix = '_h_GT'
        self.l = '_l'
        self.data = []
        self.root_dir = root_dir
        assert self._get_len_dir(root_dir+'/H_GT') == self._get_len_dir(root_dir+'/L'), 'Length H_GT is not equal to L'
        
        h_gt_file = os.listdir(root_dir+'/H_GT')
        for i in h_gt_file:
            file_name = i[:-4]
            self.data.extend(total_image(file_name,
                                         self.frames_video,
                                         self.image_ext,
                                         self.split,
                                         self.root_dir)
                             )
        
        random.shuffle(self.data)
        self.transforms = transforms
        
    def _get_len_dir(self,path):
        return len(os.listdir(path))
    
    def __getitem__(self,index):
        gt_path, l_path = self.data[index]
        gt_image = PIL.Image.open(gt_path)
        l_image  = PIL.Image.open(l_path)
        if self.transforms:
            gt_image = self.transforms(gt_image)
            l_image  = self.transforms(l_image)
        
        return l_image,gt_image
        
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    ds = YouKuSrDataset(split='val',root_dir='video_image/val')
    print(ds[10][0].show())
        
        
