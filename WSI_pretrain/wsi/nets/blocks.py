import torch.nn as nn

# 分类头
class Classify_Head(nn.Module):
    def __init__(self, backbone, backbone_dim,class_num):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone_dim,class_num)
    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x)
        return x
