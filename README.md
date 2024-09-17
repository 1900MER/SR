## This repo is a Super Resolution competition held by Youku and Alibaba
The Image restoration, including super resolution, image denosing, aims to recover from the low quality image to high quality image. 
## Usage 
```bash
$ pip install -r requirements.txt      
```
## Dataset
Put the train and val dataset in video_image. In each train,val create H_GT and L for low resolution image and ground truth image respectively.
Then 
'''bash
$ python main.py
'''
To run the training process.
## Detail
The SwinIR model here takes (270,480) size image as input and the scale factor is 4 which makes the output image size is (1080,1920). 
## Citation
```bibtex
@article{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2108.10257},
  year={2021}
}
```

