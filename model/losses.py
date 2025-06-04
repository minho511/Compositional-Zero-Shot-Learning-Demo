import torch
import torch.nn as nn
    

class ReconLoss(nn.Module):
    def __init__(self):
        super(ReconLoss, self).__init__()
        self.l1recon = nn.L1Loss(reduction='none')

    def forward(self, x, target):
        abs_diff = self.l1recon(x, target.detach())
        w = torch.pow(abs_diff, 2)
        w = w / torch.norm(w, p=2)
        weighted_loss = w.detach() * abs_diff
        loss = weighted_loss.mean()
        return loss
