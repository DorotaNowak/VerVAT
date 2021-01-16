import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        label = label.squeeze()
        loss_contrastive = torch.sum(1 / 2 * label * (label + 1) * torch.pow(euclidean_distance, 2) +
                                     1 / 2 * label * (label - 1) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
