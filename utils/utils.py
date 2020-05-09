# --------------------------------------------------------
# Written by Hamadi Chihaoui at 8:59 AM 5/8/2020 
# --------------------------------------------------------
import numpy as np
import cv2


def bbox2loc(src_bbox, des_bbox):
    """
    :param src_bbox: #shape h * w * n_anchor, 4
    :param locs: #shape  h * w * n_anchor, 4
    """
    src_h = src_bbox[:, 2] - src_bbox[:, 0]
    src_w = src_bbox[:, 3] - src_bbox[:, 1]
    des_h = src_bbox[:, 2] - des_bbox[:, 0]
    des_w = src_bbox[:, 3] - des_bbox[:, 1]
    eps = np.finfo(src_h.dtype).eps
    src_h = np.maximum(src_h, eps)
    src_w = np.maximum(src_w, eps)

    dh = np.log(des_h / src_h)
    dw = np.log(des_w / src_w)

    dy = (des_bbox[:, 0] - src_bbox[:, 0]) / src_h
    dx = (des_bbox[:, 1] - src_bbox[:, 1]) / src_w

    loc = np.concatenate([dy, dx, dh, dw], axis=-1)

    return loc

def loc2bbox(src_bbox, locs):
    """

    :param src_bbox: #shape h * w * n_anchor, 4
    :param locs: #shape  h * w * n_anchor, 4
    """
    import numpy as np
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = locs[:, 0]
    dx = locs[:, 1]
    dh = locs[:, 2]
    dw = locs[:, 3]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.concatenate([ctr_y - 0.5 * h, ctr_x - 0.5 * w, ctr_y + 0.5 * h, ctr_x + 0.5 * w], axis=-1)
    return dst_bbox

def iou(bboxes_a, bboxes_b):
    # bboxes shape N1 * 4
    # bboxes shape N2 * 4
    min_y2 = np.minimum(bboxes_a[:, None, 2], bboxes_b[:, 2])
    max_y1 = np.maximum(bboxes_a[:, None,  0], bboxes_b[:, 0])
    min_x2 = np.minimum(bboxes_a[:, None,  3], bboxes_b[:, 3])
    max_x1 = np.maximum(bboxes_a[:, None, 1], bboxes_b[:, 1]) # shape N1 * N2

    intersection_y = np.maximum(min_y2 - max_y1, 0.) # shape N1 * N2
    intersection_x = np.maximum(min_x2 - max_x1, 0.)
    intersection_area = intersection_y * intersection_x # shape N1 * N2

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1]) # shape N1
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1]) # shape N2

    # area_a_1 = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], axis=1)
    # area_b_1 = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], axis=1)

    sum_area = np.zeros((area_a.shape[0], area_b.shape[0]))
    for i in range(area_a.shape[0]):
        for j in range(area_b.shape[0]):
            sum_area[i, j] = area_a[i] + area_b[j]
    iou = intersection_area / (sum_area - intersection_area)
    return iou


def intersection_over_union(bbox_a, bboxes):
    x1_max = np.maximum(bbox_a[0], bboxes[:, 0])
    y1_max = np.maximum(bbox_a[1], bboxes[:, 1])
    x2_min = np.minimum(bbox_a[2], bboxes[:, 2])
    y2_min = np.minimum(bbox_a[3], bboxes[:, 3])

    intersection_w = np.maximum(x2_min - x1_max, 0.)
    intersection_h = np.maximum(y2_min - y1_max, 0.)
    intersection_area = intersection_w * intersection_h

    area_bbox_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_bbox_b = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    iou = intersection_area / (area_bbox_a + area_bbox_b - intersection_area)

    return iou

def non_maximum_suppression(bboxes, threshold, confidence_scores=None):
    # x1= bboxes[:, 0]
    # y1= bboxes[:, 1]
    # x2= bboxes[:, 2]
    # y2= bboxes[:, 3]
    #
    # areas = (x2 - x1) * (y2 - y1)

    # Picked bounding boxes
    picked_boxes = []
    # if confidence_scores is None :
    picked_scores = []

    # Sort by confidence score of bounding boxes
    order = np.argsort(confidence_scores)
    while order.size > 0:

        i = order[-1]
        print('i ', i)
        print('order ', order[:-1])
        print(bboxes[i])
        # Pick the bounding box with largest confidence score
        picked_boxes.append(bboxes[i])
        if confidence_scores is not None:
            picked_scores.append(confidence_scores[i])

        iou = intersection_over_union(bboxes[i], bboxes[order[:-1], :])
        keep = np.where(iou > threshold)
        order = order[keep]
    if confidence_scores is not None:
        return picked_boxes, picked_scores
    else:
        return picked_boxes


def generate_base_anchor(scales=[128, 256, 512], ratios=[0.5, 1, 2]):

    anchor_base = np.zeros(len(scales) * len(ratios), 4).reshape((len(scales), len(ratios), 4))
    for i in range(len(scales)):
        scale = scales[i]
        for j in range(len(ratios)):
            ratio = ratios[j]
            w = scale * np.sqrt(ratio)
            h = scale * np.sqrt(1. / ratio)
            anchor_base[i, j, 0] = -1 * h / 2.
            anchor_base[i, j, 1] = -1 * w / 2.
            anchor_base[i, j, 2] = h / 2.
            anchor_base[i, j, 3] = w / 2.

    return anchor_base.reshape(len(scales) * len(ratios), 4)  # shape (len(scales) * len(ratios), 4)


def generate_anchors(base_anchor, feat_width, feat_height, feat_stride):

    shift_y = np.arange(0, feat_height * feat_stride, feat_stride)
    shift_x = np.arange(0, feat_width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1) #shape (feat_height * feat_width, 4)
    anchors = shift.reshape((1, -1, 4)).transpose((1, 0, 2)) + base_anchor.reshape((1, -1, 4))
    anchors = anchors.reshape(-1, 4).astype(np.float32)

    return anchors


if __name__ == '__main__':

    bboxes_a = np.array([[20,20, 40,40], [40,40,60,60]])
    bboxes_b = np.array([[30, 30, 50, 50], [50, 50, 70, 70]])
    iou = iou(bboxes_a, bboxes_b)


    # # Image name
    # image_name = '1_684.jpg'
    #
    # # Bounding boxes
    # bounding_boxes = np.array([[187, 82, 337, 317], [150, 67, 305, 282], [246, 121, 368, 304], [200, 170, 400, 180]])
    # confidence_score = [0.9, 0.75, 0.8, 0.5]
    #
    # # Read image
    # image = cv2.imread(image_name)
    #
    # # Copy image as original
    # org = image.copy()
    #
    # # Draw parameters
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 1
    # thickness = 2
    #
    # # IoU threshold
    # threshold = 0.7
    #
    # # Draw bounding boxes and confidence score
    # for (start_x, start_y, end_x, end_y), confidence in zip(bounding_boxes, confidence_score):
    #     (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
    #     cv2.rectangle(org, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
    #     cv2.rectangle(org, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    #     cv2.putText(org, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)
    #
    # # Run non-max suppression algorithm
    # picked_boxes, picked_score = non_maximum_suppression(bounding_boxes,threshold, confidence_score)
    #
    # # Draw bounding boxes and confidence score after non-maximum supression
    # for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
    #     (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
    #     cv2.rectangle(image, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
    #     cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    #     cv2.putText(image, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)
    #
    # # Show image
    # cv2.imshow('Original', org)
    # cv2.imshow('NMS', image)
    # cv2.waitKey(0)

