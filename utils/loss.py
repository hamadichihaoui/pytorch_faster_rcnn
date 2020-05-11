# --------------------------------------------------------
# Written by Hamadi Chihaoui at 10:08 PM 5/11/2020 
# --------------------------------------------------------

from torch import nn
import torch

class SmoothL1Loss(nn.Module):
    def __init__(self, alpha=0.01):
        super(SmoothL1Loss, self).__init__()
        self.alpha = alpha


    def forward(self, gt_locs, predicted_locs):
        # locs shape [bach_size, n, 4]
        gt_locs = gt_locs.view(-1, 4).contigious()
        predicted_locs = predicted_locs.view(-1, 4).contigious()

        mean_abs_diff = torch.mean(torch.abs(gt_locs - predicted_locs))
        smooth_l1_loss = mean_abs_diff * (mean_abs_diff >= self.alpha) + mean_abs_diff ** 2 \
                         / self.alpha * (mean_abs_diff < self.alpha)


        return smooth_l1_loss.sum()




