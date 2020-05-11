# --------------------------------------------------------
# Written by Hamadi Chihaoui at 10:05 PM 5/11/2020 
# --------------------------------------------------------
import torch
from torch import nn
from models.creator_tool import ProposalTargetCreator, AnchorTargetCreator
from utils.loss import SmoothL1Loss

class FasterRCNNTrainer(nn.Module):
    def __init__(self, fasterrcnn):
        super(FasterRCNNTrainer, self).__init__()
        self.fasterrcnn = fasterrcnn

        self.ProposalTargetCreator = ProposalTargetCreator()
        self.AnchorTargetCreator = AnchorTargetCreator()
        self.optimizer = self.faster_rcnn.get_optimizer()
        self.binary_cross_entropy = nn.BCELoss()
        self.smooth_l1_loss = SmoothL1Loss()


        self.rpn_loc_loss = 0.
        self.rpn_cls_loss = 0.
        self.roi_loc_loss = 0.
        self.roi_cls_loss = 0.


    def forward(self, x, bboxes, labels):

        features = self.fasterrcnn.feature_extractor(x)

        rpn_objectness_scores, rpn_locs, rois, roi_indices, anchors = self.fasterrcnn.rpn(features, x.shape)
        gt_objectness_scores, gt_rpn_locs = self.AnchorTargetCreator(bboxes, anchors, features.shape)

        ###------------------------------------ RPN Losses ------------------------------------###
        self.rpn_loc_loss = self.smooth_l1_loss(rpn_locs, gt_rpn_locs)
        self.rpn_cls_loss = self.binary_cross_entropy(gt_objectness_scores, rpn_objectness_scores)

        ###------------------------------------------------------------------------------------###

        sampled_rois, gt_rois_locs, gt_rois_labels = self.ProposalTargetCreator(rois, bboxes, labels)

        roi_cls_locs, roi_scores = self.fasterrcnn.head(features, sampled_rois)

        ###------------------------------------ ROI Losses ------------------------------------###
        self.roi_loc_loss = 0.
        self.roi_cls_loss = self.binary_cross_entropy(gt_rois_labels, roi_scores)
        ###------------------------------------------------------------------------------------###



    def train_step(self):
        self.optimizer.zero_grad()


