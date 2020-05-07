# --------------------------------------------------------
# Written by Hamadi Chihaoui at 11:54 AM 5/7/2020
# --------------------------------------------------------

from torch import nn


class FasterRCNN(nn.Module):

    def __init__(self, feature_extractor, rpn, roi_pooling):
        super(FasterRCNN, self).__init__()
        self.feature_extractor = feature_extractor
        self. rpn = rpn
        self.roi_pooling = roi_pooling

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        rpn_proposals, rpn_locs, rpn_scores = self.rpn(feature_map)
        roi_locs, roi_scores = self.roi_pooling(feature_map, rpn_proposals)

        return roi_locs, roi_scores
