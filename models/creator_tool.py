# --------------------------------------------------------
# Written by Hamadi Chihaoui at 10:50 AM 5/9/2020 
# --------------------------------------------------------


from torch import nn
from utils.utils import *


class ProposalTargetCreator:
    def __init__(self,
                 n_sample=128,
                 pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0,
                 pos_ratio=0.25, num_classes=21):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        #self.neg_iou_thresh = neg_iou_thresh
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.pos_ratio = pos_ratio
        self.num_classes = num_classes

    def __call__(self, rois, bboxes, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals."""
        # label = np.empty((len(rois), self.num_classes), dtype=np.int32)
        # label.fill(-1)
        n_bbox, _ = bboxes.shape

        roi = np.concatenate((rois, bboxes), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou1 = iou(roi, bboxes)
        gt_assignment = iou1.argmax(axis=1)
        max_iou = iou1.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bboxes[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label


def calculate_iou(self, anchors, bboxes):
        ious = iou(anchors, bboxes)
        max_ious = np.max(ious, axis=1)  # max overlap with every anchor
        arg_max_ious = np.argmax(axis=1) # which bbox with max overlap with every anchor
        gt_arg_max_ious = np.argmax(axis=0) # which anchor with max overlap with every bbox

        return arg_max_ious, max_ious, gt_arg_max_ious





class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.
    Assigns the ground truth bounding boxes to anchors for training Region"""

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bboxes, anchors, img_size):
        """Assign ground truth supervision to sampled subset of anchors."""
        # n = self.anchors.shape[0]
        # final_matching = np.empty((n,), dtype=bool)
        # final_matching.fill(-1)
        # for bb in bbox:
        #     iou = intersection_over_union(bb, self.anchors)
        #     pos_matching = np.where(iou > 0.7)
        #     final_matching[pos_matching] = 1
        #     neg_matching = np.where(iou < 0.3)
        #     final_matching[pos_matching] = np.maximum(final_matching[pos_matching], 0)
        #
        #     # if there is not any matching, force matching to nearest anchor_box
        #     if len(pos_matching) == 0:
        #         order = iou.argsort()[::-1]
        #         final_matching[order[0]] = 1
        indices_inside = self.filter_anchors_that_does_not_lie_completely_inside_the_image(anchors, img_size)
        anchors = anchors[indices_inside]
        arg_max_ious, label = self.create_label(anchors, bboxes, indices_inside)
        loc = bbox2loc(anchors, bboxes[arg_max_ious])
        n_anchor = len(anchors)

        # map up to original set of anchors
        label = self.unmap(label, n_anchor, indices_inside, fill=-1)
        loc = self.unmap(loc, n_anchor, indices_inside, fill=0)

        return label, loc

    def unmap(self, data,  count, index, fill=0):
        if len(data.shape) == 1:
            out = np.empty((count, ), dtype= data.dtype)
            out.fill(fill)
            out[index] = data
        else:
            out = np.empty((count, data.shape[1]), dtype=data.dtype)
            out.fill(fill)
            out[index, :] = data

        return out

    def create_label(self, anchors, bboxes, indices_inside):
        label = np.empty((len(indices_inside), ),  dtype=np.int32)
        label.fill(-1)
        arg_max_ious, max_ious, gt_arg_max_ious = self.calculate_iou(anchors, bboxes)
        label[max_ious >= self.pos_iou_thresh] = 1
        label[max_ious < self.neg_iou_thresh] = 0
        label[gt_arg_max_ious] = 1

        n_pos_to_sample = int(self.n_sample * self.pos_ratio)

        actual_pos = np.where(label == 1)
        if len(actual_pos) > n_pos_to_sample:
            disable_index = np.random.choice(
                actual_pos, size=(len(actual_pos) - n_pos_to_sample), replace=False)
            label[disable_index] = -1

        n_neg_to_sample = int(self.n_sample * (1-self.pos_ratio))

        actual_neg = np.where(label == 0)
        if len(actual_pos) > n_neg_to_sample:
            disable_index = np.random.choice(
                actual_neg, size=(len(actual_neg) - n_neg_to_sample), replace=False)
            label[disable_index] = -1

        return arg_max_ious, label


    def calculate_iou(self, anchors, bboxes):
        ious = iou(anchors, bboxes)
        max_ious = np.max(ious, axis=1)  # max overlap with every anchor
        arg_max_ious = np.argmax(axis=1) # which bbox with max overlap with every anchor
        gt_arg_max_ious = np.argmax(axis=0) # which anchor with max overlap with every bbox

        return arg_max_ious, max_ious, gt_arg_max_ious

    def filter_anchors_that_does_not_lie_completely_inside_the_image(self, anchors, img_size):
        # anchors shape:[h * w * n_anchor,4]
        indices_inside = np.where(anchors[:, 0] >= 0 and
        anchors[:, 2] <= img_size[0] and
        anchors[:, 1] >= 0 and
        anchors[:, 3] <= img_size[1])

        return indices_inside



class ProposalGenerator:
    '''
     generate most relevant (predicted) proposals (RPN)
    '''

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def generate_proposals_single_image(self, localization_scores_single_image, rpn_fg_scores_single_image, anchors,
                                        img_size):

        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

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

        # convert to relative dimensions
        roi[:, slice(0, 4, 2)] /= img_size[0]
        roi[:, slice(1, 4, 2)] /= img_size[1]

        return roi
