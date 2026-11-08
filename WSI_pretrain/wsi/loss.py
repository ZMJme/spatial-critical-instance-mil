import torch
import torch.nn as nn
from .utils import *

# 交叉熵损失，weight为不同类别损失权重计算时会自动进行归一化处理
class CE_Loss(nn.Module):
    def __init__(self,weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self,predict,label):
        predict = torch.sigmoid(predict)
        # view之前检查维度顺序！！TMD
        c = predict.size(1)
        if len(predict.shape) == 4:
            # 二维图像 b,c,w,h  -->  b,w,h,c
            predict = predict.permute(0,2,3,1)
        else:
            # 三维图像 b,c,w,d,h  -->  b,w,h,d,c
            predict = predict.permute(0,2,3,4,1)
            
        predict = predict.contiguous().view((-1,c))
        label = label.view(-1)

        return self.ce_loss(predict,label)



# Dice-交叉熵混合损失
class Dice_CE_Loss(nn.Module):
    def __init__(self, n_class,weight=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__EPS__ = 1e-9
        self.n_class = n_class
        self.ce_loss = CE_Loss(weight=weight)
        
    def dice_loss(self,pred,label):
        c = pred.size(1)
        if len(pred.shape) == 4:
            # 二维图像 b,c,w,h  -->  b,w,h,c
            pred = pred.permute(0,2,3,1)
        else:
            # 三维图像 b,c,w,h,d  -->  b,w,h,d,c
            pred = pred.permute(0,2,3,4,1)
        pred = pred.contiguous().view((-1,c))
        label = label.view(-1)

        pred = torch.sigmoid(pred)
        label = torch.eye(self.n_class).to(label.device)[label]
        
        return 1-2*(pred*label).sum()/(pred.sum()+label.sum()+self.__EPS__)
    
    def forward(self,predict,label):
        return self.dice_loss(predict,label)+self.ce_loss(predict,label)



# Focal Loss: alpha类型为tensor,是类别权重
class FocalLoss(nn.Module):
    def __init__(self,class_num,gamma=2,alpha=None,reduction="mean") -> None:
        super().__init__()
        if alpha is None:
            self.alpha = torch.ones((class_num,1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num
    
    def forward(self,predict,target):
        pt = torch.nn.functional.softmax(predict,dim=1)
        class_mask = torch.nn.functional.one_hot(target,self.class_num)
        ids = target.view(-1,1)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (pt*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        loss = -alpha*(torch.pow(1-probs,self.gamma))*log_p
        if self.reduction=="mean":
            loss = loss.mean()
        elif self.reduction=="sum":
            loss = loss.sum()
        return loss



# MAE均方误差损失
class MAE_MSE_Loss(nn.Module):
    def __init__(self,patch_size):
        super().__init__()
        self.patch_size = patch_size
    def patchify(self, imgs):
        # (B, C, H, W) -> (B, L, patch_size**2 *C)
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    def unpatchify(self, x):
        # (N, L, patch_size**2 *C) -> (N, 3, H, W)
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    def forward(self,pred,x,mask):
        # x: b,c,h,w    pred: b,l,h*w*c   mask: l
        target = self.patchify(x) # b,l,h*w*c
        mask = torch.tensor(mask).to(x.device)
        mask = ~mask
        mask = torch.ones_like(mask)
        # MSE_loss
        loss = (pred-target)**2
        loss = loss.mean(dim=-1) # b,l
        loss = (loss*mask).sum()/mask.sum()
        return loss