# --------------------------------------------------------
# Written by Hamadi Chihaoui at 1:20 PM 5/8/2020 
# --------------------------------------------------------
from torch import nn
from models.faster_rcnn import FasterRCNN
from models.backbones import vgg_extractor, vgg_head
from models.roi_pooling import RoIPooling2D
from models.rpn import RPN
import numpy as np
import torch


class VGG_FasterRCNN(FasterRCNN):
    def __init__(self, n_classes, roi_size=7):
        self.feature_extractor = vgg_extractor()
        self.rpn = RPN()
        self.head = VGG16RoIHead(n_classes, roi_size)
        super(VGG_FasterRCNN, self).__init__(self.feature_extractor, self.rpn, self.head)



class VGG16RoIHead(nn.Module):

    def __init__(self, n_class, roi_size):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()
        self.classifier = vgg_head()
        self.cls_loc = nn.Linear(4096,  n_class * 4)
        self.score = nn.Linear(4096,  n_class)
        self.n_class = n_class
        self.roi_size = roi_size
        self.roi = RoIPooling2D()

    def forward(self, x, rois):
        #TODO remove indices as indice refers to image index in a batch while we only support batch size=1
        # in case roi_indices is  ndarray
        #roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        rois = rois.contiguous()
        # indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # # NOTE: important: yx->xy
        # xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        # indices_and_rois = xy_indices_and_rois.contiguous()
        pool = self.roi(x, rois) #indices_and_rois
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor