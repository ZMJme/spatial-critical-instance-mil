"""
openslide安装

sudo apt-get install openslide-tools
conda install -c conda-forge openslide==3.4.1
pip install openslide-python
"""


import openslide
from lxml import etree
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm
import os
import torch
import networkx as nx
import torch.nn.functional as F
from torchvision import transforms


__WSI_TYPE__ = "Camelyon16"
# __WSI_TYPE__ = "BACH"



# WSI处理类
class WSI:
    def __init__(self,wsi_path):
        self.wsi = openslide.OpenSlide(wsi_path)
        # 等级数
        self.level_count = self.wsi.level_count
        # 各等级对应的下采样倍数
        self.level_downsamples = self.wsi.level_downsamples
        # 各等级对应的尺寸 (w,h)
        self.level_dimensions = self.wsi.level_dimensions
    
    # 获取切片
    def read_region(self,
                    location, # 左上顶点坐标(h,w)
                    level, # wsi等级
                    size # 尺寸(h,w)
                    ):
        h,w = location
        # 必须是对应0级图下坐标
        p = int(int(self.level_downsamples[level])/int(self.level_downsamples[0]))
        w,h = w*p,h*p
        return np.array(self.wsi.read_region((w,h),level,size).convert("RGB"))

    # 获取略缩图
    def get_image(self,level):
        return np.array(self.wsi.read_region((0,0),level,self.level_dimensions[level]).convert("RGB"))
    
    # 解析xml文件 p:下采样倍数
    def parse_xml(self,xml_path,p=1):
        if __WSI_TYPE__ == "BACH":
            region_xy_list = []
            class_list = []
            tree = etree.parse(xml_path)
            for region in tree.findall(".//Region"):
                # 各区域所属类别
                r_class = region.attrib["Text"]
                if r_class=="":
                    r_class = region.findall(".//Attribute")[0].attrib["Value"]
                class_list.append(r_class)
                # 各区域多边形顶点
                vertex_list = []
                for vertex in region.findall(".//Vertex"):
                    vertex_list.append([int(vertex.attrib["X"])/p,int(vertex.attrib["Y"])/p])
                region_xy_list.append(np.array(vertex_list,np.int32))
            return region_xy_list, class_list
        
        elif __WSI_TYPE__ == "Camelyon16":
            region_xy_list = []
            class_list = []
            tree = etree.parse(xml_path)
            for region in tree.findall(".//Annotation"):
                # 各区域所属类别
                class_list.append(region.attrib["PartOfGroup"])
                # 各区域多边形顶点
                vertex_list = []
                for vertex in region.findall(".//Coordinate"):
                    vertex_list.append([int(float(vertex.attrib["X"]))/p,int(float(vertex.attrib["Y"]))/p])
                region_xy_list.append(np.array(vertex_list,np.int32))

            return region_xy_list, class_list
    
    
    # 根据xml生成mask,返回mask与类别标签字典    level:缩放等级
    def get_mask(self,xml_path,level):
        level = level if level<self.level_count else self.level_count-1
        # 解析xml文件
        p = self.level_downsamples[level]
        region_xy_list, class_list = self.parse_xml(xml_path,p)
        # 生成mask
        (w,h) = self.level_dimensions[level]
        class_lebel_dic = {c:i+1 for i,c in enumerate(set(class_list))}

        if __WSI_TYPE__ == "Camelyon16":
            # Camelyon16类别_2表示去除
            class_lebel_dic["_2"]=0

        mask = np.zeros((h,w))
        for i,r in enumerate(region_xy_list):
            mask = cv.fillPoly(mask,[r],(class_lebel_dic[class_list[i]]))
        return mask,class_lebel_dic
    
    # 检测组织前景mask，返回组织前景
    def get_tissue_mask(self,
                        level_base = None,
                        ):
        
        level_base = self.level_count-1 if level_base is None else level_base
        # 最高级别略缩图
        img = self.get_image(level_base)
        # hsv阈值分割
        hsv = cv.cvtColor(img,cv.COLOR_RGB2HSV)
        _,mask = cv.threshold(hsv[:,:,1],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)        
        # 形态学运算去除毛刺、空洞等
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, (15, 15))
        mask[mask>0]=1
        return mask

    # 获取具有前景的patch坐标
    def get_tissue_patches(self,
                           level, # 在哪个等级下进行切割
                           level_base = None,
                           patch_size=512, # 切片大小
                           r = 0.3, # 前景占比阈值，滑动窗口中前景占比大于该阈值则保留
                           ):
        # 最高级别略缩图及对应的下采样倍数
        level_base = self.level_count-1 if level_base is None else level_base
        mask = self.get_tissue_mask(level_base)
        # 最高级别略缩图对应当前级别略缩图的下采样倍数
        p = int(self.level_downsamples[level_base] / self.level_downsamples[level])
        # 滑动窗口检测含前景组织区域
        rects = []
        window_size = max(1,int(patch_size/p))
        (h,w) = mask.shape
        for y in range(0, h, window_size):
            for x in range(0, w, window_size):
                rec = mask[y:y + window_size, x:x + window_size]
                if (rec>0).sum() > (rec.shape[0]*rec.shape[1])*r:
                    # 切片目标级别的略缩图上区域左上顶点坐标 (h,w)
                    x_start,y_start = int(x*p),int(y*p)
                    x_end,y_end = int((x+1)*p),int((y+1)*p)
                    for x_i in range(x_start,x_end,patch_size):
                        for y_i in range(y_start,y_end,patch_size):
                            rects.append([y_i,x_i])
        return level,rects,patch_size
    
    # 切片保存任务
    def __cut_save(self,h,w,level,size,save_path):
        img = self.read_region((h,w),level,size)
        cv.imwrite(os.path.join(save_path,f"{h}_{w}.png"), img)
    # 大图切片,只切有前景的区域
    def patch_cut(self,
                  level, # 等级
                  rects, # 左上顶点坐标列表
                  patch_size, #切割尺寸
                  save_path, # 保存路径
                  ):
        p = int(self.level_downsamples[level])
        # 多线程切割
        threadpool = ThreadPoolExecutor(20)
        futures = [threadpool.submit(self.__cut_save,int(h*p),int(w*p),level,(patch_size,patch_size),save_path) for h,w in rects]
        progress_bar = tqdm(total=len(rects))
        for future in as_completed(futures):
            # 获取任务结果
            result = future.result()
            # 更新进度条
            progress_bar.update(1)
        threadpool.shutdown()




class WSI_Graph(WSI):
    def __init__(self, wsi_path, label_path, label_base_level=5):
        super().__init__(wsi_path)
        # 标注处理
        self.label_base_level = label_base_level if label_base_level<self.level_count else self.level_count-1
        if label_path is not None:
            # 有标注文件
            self.label_mask,self.class_lebel_dic = self.get_mask(label_path,label_base_level)
        else:
            # 没有标注文件
            (w,h) = self.level_dimensions[label_base_level]
            self.label_mask = np.zeros((h,w))
            self.class_lebel_dic = {}
        # 标注中是否有前景
        self.label_is_empty = (self.label_mask.sum()==0)

        # 标准化
        self.normalize = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    # 获取切片标签
    def read_region_label(self,
                    location, # 左上顶点坐标(h,w)
                    level, # wsi等级
                    size, # 尺寸(h,w)
                    r=0.5
                    ):
        # 采样级别相对于mask所在级别的放大倍率
        p = int(self.level_downsamples[self.label_base_level] / self.level_downsamples[level])

        if __WSI_TYPE__ == "BACH":
            # # BACH 标签判定策略：看中心点像素类别判断
            h,w = location
            w, h, size =  int(w/p), int(h/p), int(size/p/2)
            return self.label_mask[h+size, w+size]

        elif __WSI_TYPE__ == "Camelyon16":
            # Camelyon16 标签判定策略：面积占比大于阈值
            h,w = location
            w, h, size =  int(w/p), int(h/p), int(size/p)
            m = self.label_mask[h:h+size, w:w+size]
            if (not self.label_is_empty) and ((m>0).sum()>(size**2)*r):
                return 1
            else:
                return 0
        

    
    # 获得图结构
    def get_graph(self,level,level_base,patch_size=256,r=0.5):
        level_base = level_base if level_base<self.level_count else self.level_count-1
        _,rects,_ = self.get_tissue_patches(level,level_base,patch_size,r)
        edge_list = []
        for i in range(len(rects)):
            for j in range(i):
                (y_i,x_i),(y_j,x_j) = rects[i],rects[j]
                if abs(y_i-y_j)<=patch_size and abs(x_i-x_j)<=patch_size:
                    edge_list.append([i,j])
        G = nx.Graph()
        for i,(y,x) in enumerate(rects):
            G.add_node(i,x=x,y=y,label=self.read_region_label((y,x),level,patch_size))
        G.add_edges_from(edge_list)
        return G
    
    # 获取指定区域为tensor
    def read_region_tensor(self, location_list, level, patch_size):
        size = (patch_size,patch_size)
        img_list = []
        for location in location_list:
            img = super().read_region(location, level, size)
            img = torch.tensor(img.astype(np.float64)).permute(2,0,1).unsqueeze(0)
            img = self.normalize(img)
            img_list.append(img)
        img = torch.concat(img_list,0)
        return img

