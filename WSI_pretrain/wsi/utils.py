import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import networkx as nx
from scipy.spatial import Delaunay
import logging
import os
import pickle

__EPS__ = 1e-9

# IoU
def get_iou(pred,mask,class_num):
    ious = []
    pred = pred.view(-1)
    mask = mask.view(-1)
    for i in range(class_num):
        p_i = pred==i
        m_i = mask==i
        iou = ((p_i&m_i).sum()+__EPS__)/((p_i|m_i).sum()+__EPS__)
        ious.append(iou)
    return ious

# Dice
def get_dice(pred,mask,class_num):
    dices = []
    pred = pred.view(-1)
    mask = mask.view(-1)
    for i in range(class_num):
        p_i = pred==i
        m_i = mask==i
        dice = (2*(p_i&m_i).sum()+__EPS__)/((p_i).sum()+(m_i).sum()+__EPS__)
        dices.append(dice)
    return dices

# 混淆矩阵
def get_confusion_matrix(pred,mask,labels=None):
    pred = pred.view(-1)
    mask = mask.view(-1)
    return confusion_matrix(mask,pred,labels=labels)

# 图片转onehot向量（通道对应类别标签）
def one_hot(imgs,num_class):
    b,c,w,h = imgs.shape
    assert c==1
    imgs = imgs.view(-1)
    imgs = torch.eye(num_class).to(imgs.device)[imgs]
    imgs = torch.reshape(imgs,[b,w,h,num_class])
    imgs = imgs.permute(0,3,1,2)
    return imgs

# pickle加载对象
def pickle_load(fpath):
    with open(fpath,"rb")as fp:
        data = pickle.load(fp)
    return data

# pickle保存对象
def pickle_dump(obj,fpath):
    with open(fpath,"wb")as fp:
        pickle.dump(obj,fp)

# 计算模型参数量
def model_param_count(model):
    return sum([p.numel() for p in model.parameters()])


# 展示分割结果    img: 原图像(3,512,512); mask: 原mask图像(512,512); pred_mask: 分割结果(512,512) 
def result_show(img,mask,pred_mask):
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.title("img")
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.title("mask")
    plt.imshow(mask,cmap="RdYlBu_r")
    plt.subplot(1,3,3)
    plt.title("pred")
    plt.imshow(pred_mask,cmap="RdYlBu_r")
    plt.show()
    
    img = img.copy()
    contours, _ = cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(img,contours,-1,(0,255,0),2)
    contours, _ = cv.findContours(pred_mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(img,contours,-1,(255,0,0),2)
    plt.figure()
    plt.imshow(img)
    plt.show()

# 展示图象 img(H,W,C)
def img_show(img,cmap=None):
    plt.figure()
    plt.imshow(img,cmap=cmap)
    plt.show()

# 展示图结构
def network_show(g,figsize=(12,12)):
    nx.write_edgelist(g, path="grid.edgelist", delimiter=":")
    # read edgelist from grid.edgelist
    H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

    pos = nx.spring_layout(H)

    nx.draw_networkx(H,pos,arrows=False,node_size=5,with_labels=False)


# 根据点集计算德劳内三角形
def get_delaunay(points):
    result=[]
    tri = Delaunay(points)
    for (i1,i2,i3) in tri.simplices:
        p1,p2,p3 = points[i1],points[i2],points[i3]
        result.append([p1,p2,p3])
    return np.array(result)


# 获取一个logger
def get_logger(save_path):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logger.handlers=[]
    fh = logging.FileHandler(os.path.join(save_path,'log.log'),"w")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger



# 持久化对象
def pickle_dump(obj,f_path):
    with open(f_path,"wb")as fp:
        pickle.dump(obj,fp)



# 读取对象
def pickle_load(f_path):
    try:
        with open(f_path,"rb")as fp:
            return pickle.load(fp)
    except:
        return None
