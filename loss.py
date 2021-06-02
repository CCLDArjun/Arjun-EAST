import torch
from monai.losses.dice import DiceLoss

class EASTLoss(torch.nn.Module):
    def __init__(self):
        super(EASTLoss, self).__init__()
        self.dice_loss = DiceLoss(sigmoid=True)

    def forward(self, gt_score, pred_score, gt_rbox, pred_rbox):
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_rbox) * 0

        score_loss = self.dice_loss(pred_score, gt_score)
        geo_loss = # need to implement -> rbox_loss(pred_rbox[0:4], gt_rbox[0:4])
        angle_loss = 1 - torch.cos(pred_rbox[4] - gt_rbox[4])
        geo_loss = torch.sum(angle_loss, geo_loss)

        return torch.sum(geo_loss, score_loss)
