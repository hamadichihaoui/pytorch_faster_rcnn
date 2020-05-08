# --------------------------------------------------------
# Written by Hamadi Chihaoui at 11:54 AM 5/7/2020
# --------------------------------------------------------

from torch import nn


class FasterRCNN(nn.Module):

    def __init__(self, feature_extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        self.feature_extractor = feature_extractor
        self. rpn = rpn
        self.head = head

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        rpn_objectness_scores, rpn_localization,  rois, roi_indices, anchors = self.rpn(feature_map)
        roi_locs, roi_scores = self.head(feature_map, rois)

        return roi_locs, roi_scores
