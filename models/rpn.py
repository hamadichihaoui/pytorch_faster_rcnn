# --------------------------------------------------------
# Written by Hamadi Chihaoui at 11:54 AM 5/7/2020
# --------------------------------------------------------

from torch import nn

class RPN(nn.Module):

    def __init__(self, features_channels=512, intermediate_channels=512, n_anchor=9):
        super.RPN.__init__()
        self.features_channels = features_channels
        self.intermediate_channels = intermediate_channels
        self.n_anchor = n_anchor
        self.conv1 = nn.Conv2d(in_channels=self.features_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_obj = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=2 * self.n_anchor, kernel_size=1, stride=1, padding=1)
        self.conv2_loc = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=4 * self.n_anchor, kernel_size=1, stride=1, padding=1)


    def _generate_proposals(self):
        raise NotImplemented

    def forward(self, x):
        h = self.conv1(x)
        objectness_scores = self.conv2_obj(h)
        localization_scores = self.conv2_loc(h)
        proposals = self._generate_proposals()

        return proposals, objectness_scores, localization_scores



