# --------------------------------------------------------
# Written by Hamadi Chihaoui at 11:54 AM 5/7/2020
# --------------------------------------------------------

import torch
from torch import nn
from utils.config import opt

class FasterRCNN(nn.Module):

    def __init__(self, feature_extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        self.feature_extractor = feature_extractor
        self. rpn = rpn
        self.head = head

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        rpn_objectness_scores, rpn_localization,  rois, roi_indices, anchors = self.rpn(feature_map, feature_map.shape)
        roi_locs, roi_scores = self.head(feature_map, rois)

        return roi_locs, roi_scores

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer
