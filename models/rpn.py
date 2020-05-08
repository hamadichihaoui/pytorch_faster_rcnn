# --------------------------------------------------------
# Written by Hamadi Chihaoui at 11:54 AM 5/7/2020
# --------------------------------------------------------

from torch.nn import functional as F
from torch import nn
from utils.utils import *


class RPN(nn.Module):

    def __init__(self, features_channels=512, intermediate_channels=512, n_anchor=9, nms_thresh=0.7):
        super(RPN, self).__init__()
        self.features_channels = features_channels
        self.intermediate_channels = intermediate_channels
        self.n_anchor = n_anchor
        self.nms_thresh = nms_thresh
        self.base_anchor = generate_base_anchor()
        self.conv1 = nn.Conv2d(in_channels=self.features_channels, out_channels=self.intermediate_channels,
                               kernel_size=3, stride=1, padding=1)
        self.objectness = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=2 * self.n_anchor,
                                    kernel_size=1, stride=1, padding=1)
        self.localization = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=4 * self.n_anchor,
                                      kernel_size=1, stride=1, padding=1)

    def _generate_proposals_single_image(self, localization_scores_single_image, rpn_fg_scores_single_image, anchors, img_size, n_post_nms=300):
        n_pre_nms = 6000
        roi = loc2bbox(localization_scores_single_image, anchors)

        # clip predicted bbox to image
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # remove/filter bbox less than minimum size
        min_size = self.min_size
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        rpn_fg_scores_single_image = rpn_fg_scores_single_image[keep]

        # sort all pair (proposal, score) by score from highest to lowest
        order = rpn_fg_scores_single_image.ravel().argsort()[::-1]

        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        roi = non_maximum_suppression(roi, self.nms_thresh)
        if n_post_nms > 0:
            roi = roi[:n_post_nms]

        return roi

    def _generate_proposals(self, localization_scores, rpn_fg_scores, anchors, img_size, n_post_nms=300):
        """

        :param localization_scores: #shape b, h * w * n_anchor, 4
        :param rpn_fg_scores: #shape b, h * w * n_anchor
        :param anchors: #shape h * w * n_anchor, 4
        """

        localization_scores = localization_scores.cpu().data.numpy()
        rpn_fg_scores = rpn_fg_scores.cpu().data.numpy()
        rois = []
        roi_indices = []
        for i in range(rpn_fg_scores.shape[0]):
            roi = self._generate_proposals_single_image(localization_scores[i], rpn_fg_scores[i], anchors, img_size, n_post_nms)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        return rois, roi_indices

    def forward(self, x, img_size):
        b, c, h, w = x.size()
        anchors = generate_anchors(self.base_anchor, h, w, feat_stride=16)
        n_anchor = anchors.shape[0] // (h * w)
        features = self.conv1(x)
        rpn_objectness_scores = self.objectness(features)
        rpn_objectness_scores = rpn_objectness_scores.permute(0, 2, 3, 1)
        probs = F.softmax(rpn_objectness_scores.view(b, h, w, n_anchor, 2), dim=4)
        rpn_fg_scores = probs[:, :, :, :, 1]
        rpn_fg_scores = rpn_fg_scores.view(b, -1)
        rpn_objectness_scores = rpn_objectness_scores.view(b, -1, 2)
        rpn_localization = self.localization(features)
        rpn_localization = rpn_localization.permute(0, 2, 3, 1).contiguous().view(b, -1, 4)
        rois, roi_indices = self._generate_proposals(rpn_localization, rpn_fg_scores, anchors, img_size)

        return rpn_objectness_scores, rpn_localization,  rois, roi_indices, anchors



if __name__ == '__main__':
    rpn = RPN()
    #rpn.generate_anchors(60,40,16)