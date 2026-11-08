import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import functional as tvF
from .wsi import WSI
import numpy as np

# WSI感兴趣区域分割
class WSIROI_Dataset(Dataset):
    def __init__(self,
                 data_path,
                 resize = None,
                 crop_size=None
                 ):
        super().__init__()
        self.img_path = os.path.join(data_path,"imgs")
        self.mask_path = os.path.join(data_path,"masks")
        self.file_list = os.listdir(self.img_path)
        self.crop_size = crop_size
        self.resize = resize
    def __len__(self):
        return len(self.file_list)
    
    def get_img_mask(self,f):
        img = read_image(os.path.join(self.img_path,f))
        mask = read_image(os.path.join(self.mask_path,f))
        # 更改尺寸
        if self.resize is not None:
            img = tvF.resize(img,self.resize,antialias=True)
            mask = tvF.resize(mask,self.resize,antialias=True)

        # 随机剪裁
        if self.crop_size is not None:
            crop_params = transforms.RandomCrop.get_params(img, self.crop_size)
            img = tvF.crop(img,*crop_params,antialias=True)
            mask = tvF.crop(mask,*crop_params,antialias=True)

        # 归一化
        img = img/255
        mask = mask/255

        return img,mask
    
    def __getitem__(self, index):
        f = self.file_list[index]
        img,mask = self.get_img_mask(f)
        return img.float(),mask.long()


# 分类数据集载入
class Patch_Classify_Dataset(Dataset):
    def __init__(self,
                 data_path,
                 resize = None,
                 crop_size=None
                 ):
        super().__init__()
        # 根路径
        self.path = data_path
        # 解析路径
        dir_walk = list(os.walk(self.path))
        # 类别-id字典
        self.class_id_dic = {c:i for i,c in enumerate(dir_walk[0][1])}
        # [文件名,类别]列表
        self.data_list = []
        for idx,(p,_,fl) in enumerate(dir_walk[1:]):
            self.data_list+=[[os.path.join(p,f),idx] for f in fl]

        self.crop_size = crop_size
        self.resize = resize

        self.normalize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    def __len__(self):
        return len(self.data_list)
    
    def get_img(self,f_path):
        img = read_image(f_path)
        img = self.normalize(img)

        # 更改尺寸
        if self.resize is not None:
            img = tvF.resize(img,self.resize,antialias=True)

        # 随机剪裁
        if self.crop_size is not None:
            crop_params = transforms.RandomCrop.get_params(img, self.crop_size)
            img = tvF.crop(img,*crop_params)
            
        return img
    
    def __getitem__(self, index):
        f_path,label = self.data_list[index]
        img = self.get_img(f_path)
        label = torch.tensor(label).long()
        return img,label







# 单张WSI随机切片数据集，通过level_list可获得多种放大倍率的图像，预训练任务可继承该类重写__getitem__方法
class WSI_Slice_DataSet(Dataset):
    def __init__(self,
                 wsi_path,
                 patch_size=256,
                 level_list=[0],
                 level_base=None,
                 max_data_num=1000
                 ):
        super().__init__()
        self.wsi = WSI(wsi_path)
        self.patch_size = patch_size
        self.level_list = level_list
        self.level_rects = {}
        for level in level_list:
            level, recs, _ = self.wsi.get_tissue_patches(level,level_base,patch_size=patch_size,r=0.5)
            self.level_rects[level] = recs
        self.max_data_num = max_data_num


    def get_data(self,level,index=None):
        if index is None:
            index = np.random.randint(len(self.level_rects[level]))
        location = self.level_rects[level][index]
        img = self.wsi.read_region(location,level,(self.patch_size,self.patch_size))
        return img

    def __len__(self):
        return self.max_data_num
    
    def __getitem__(self, index):
        level = np.random.choice(self.level_list)
        data = self.get_data(level)
        # 进一步过滤无意义图片
        while data.std()<30:
            data = self.get_data(level)
        return level,data
    

class DINO_DataSet(WSI_Slice_DataSet):
    def __init__(self, wsi_path, level_list=[0], patch_size=256, level_base=None, max_data_num=1000,
                 global_crop_size = 224,
                 loacl_crop_size = 96,
                 global_nums = 2,
                 local_nums = 8
                 ):
        super().__init__(wsi_path, patch_size, level_list, level_base, max_data_num)
        # 数据增强
        augument = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2)
        ])
        # 标准化
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 全局视野
        self.global_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(global_crop_size),
            augument,
            normalize
        ])
        # 局部视野
        self.local_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(loacl_crop_size),
            augument,
            normalize
        ])

        self.global_nums = global_nums
        self.local_nums = local_nums

    def __getitem__(self, index):
        level,img = super().__getitem__(index)
        img_list=[]
        for _ in range(self.global_nums):
            img_list.append(self.global_trans(img))
        for _ in range(self.local_nums):
            img_list.append(self.local_trans(img))

        return img_list