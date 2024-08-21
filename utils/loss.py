import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class MultiPosConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, outputs):
        feats = outputs['feats']
        labels = outputs['labels']

        device = torch.device('cuda' if feats.is_cuda else 'cpu')

        feats = F.normalize(feats, dim=-1, p=2)
        local_batch_size = feats.size(0)

        if dist.is_initialized():
            all_feats = torch.cat([torch.zeros_like(feats) for _ in range(dist.get_world_size())], dim=0).to(device)
            dist.all_gather([all_feats], feats)
            all_labels = torch.cat([torch.zeros_like(labels) for _ in range(dist.get_world_size())], dim=0).to(device)
            dist.all_gather([all_labels], labels)
        else:
            all_feats = feats
            all_labels = labels

        if local_batch_size != self.last_local_batch_size:
            mask = torch.eq(labels.view(-1, 1), all_labels.contiguous().view(1, -1)).float().to(device)
            self.logits_mask = torch.ones_like(mask)
            if dist.is_initialized():
                self.logits_mask.scatter_(
                    1,
                    torch.arange(mask.shape[0]).view(-1, 1).to(device) + local_batch_size * dist.get_rank(),
                    0
                )

            self.last_local_batch_size = local_batch_size
            self.mask = mask * self.logits_mask

        mask = self.mask

        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        logits = logits - logits.max(dim=1, keepdim=True)[0]

        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = -torch.sum(p * F.log_softmax(logits, dim=1), dim=1).mean()

        return {'loss': loss, 'image_loss': loss}

