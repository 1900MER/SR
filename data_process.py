import glob
import os

import ffmpeg

from utils import gt_l_path, read_img, timer
def video2img(dir_video, dir_imgs) -> None:
    stream = ffmpeg.input(dir_video)
    stream = ffmpeg.output(stream, dir_imgs)
    try:
        ffmpeg.run(stream)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout)
        print('stderr:', e.stderr)
        raise e
    
   
def return_path(video_image = None,gt_l : str = None):
    try:
        return gt_l_path[video_image][gt_l]
    except NameError as e:
        'gt_l must be in [gt,l]'
        
@timer
def convert_video_image_all(patial_frac:float = None, gt_l:str = None) -> None:
    root_path = return_path(video_image='video',gt_l = gt_l)
    paths  = os.listdir( root_path )
    if patial_frac:
        paths = paths[:len(paths)*patial_frac]
    for path in paths:
        one_path = os.path.join(root_path,path)
        for video in os.listdir(one_path):
            video_path = os.path.join(one_path,video)
            path_to_store_image = os.path.join(return_path('image',gt_l),video[:-4])
            if not os.path.exists(path_to_store_image):
                os.mkdir(path_to_store_image)
            video2img(video_path,path_to_store_image+'/image_%04d.jpg')
            

            

if __name__ == '__main__':
    convert_video_image_all(patial_frac=1,gt_l='l')
    #read_img('/Users/haowang/Desktop/YouKu-VESR/video_image/image_0007.jpg')
#    h_img  = read_img('/Users/haowang/Desktop/YouKu-VESR/video_image/L/Youku_00000_l/image_0001.jpg')
    # print(h_img.shape)