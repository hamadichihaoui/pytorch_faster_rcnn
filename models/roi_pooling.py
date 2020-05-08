# --------------------------------------------------------
# Written by Hamadi Chihaoui at 11:54 AM 5/7/2020
# --------------------------------------------------------

from torch import nn
import torch

class RoIPooling2D(nn.Module):

    def __init__(self, outh=7, outw=7):
        super(RoIPooling2D, self).__init__()
        self.outh = outh
        self.outw = outw
        self.spatial_scale
        self.adap_avg_pool = nn.AdaptiveAvgPool2d((outh, outw))

    @staticmethod
    def _pool_roi(self, feature_map, roi):
        """ Applies ROI Pooling to a single image and a single ROI
        """
        # Compute the region of interest
        feature_map_height = int(feature_map.size[0])
        feature_map_width = int(feature_map.size[1])

        h_start = feature_map_height * roi[0]
        h_start = h_start.to(dtype= torch.int32)
        w_start = feature_map_width * roi[1]
        w_start = w_start.to(dtype=torch.int32)
        h_end = feature_map_height * roi[2]
        h_end = h_end.to(dtype=torch.int32)
        w_end = feature_map_width * roi[3]
        w_end = w_end.to(dtype=torch.int32)

        region = feature_map[h_start:h_end, w_start:w_end, :]
        region = self.adap_avg_pool(region)
        return region# torch tensor

    def forward(self, x, rois, spatial_scale):
        regions = []
        for roi in rois:
            region = self._pool_roi(x, roi)
            regions.append(region)
        out = torch.cat(*regions, dim=0)
        return out


