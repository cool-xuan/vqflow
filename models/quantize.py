import torch
from torch import nn
from torch.nn import functional as F

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, thresh=1e-6):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.thresh = thresh

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        input = input.permute(0, 2, 3, 1)
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        quantize = quantize.permute(0, 3, 1, 2).contiguous()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
    
    def reAssign(self, dist):
        _embed = self.embed.transpose(0, 1).clone().detach()
        dist = (dist / dist.sum()).detach()

        neverAssignedLoc = dist < self.thresh
        totalNeverAssigned = int(neverAssignedLoc.sum())
        # More than half are never assigned
        if totalNeverAssigned > self.n_embed // 2:
            mask = torch.zeros((totalNeverAssigned, ), device=self.embed.device)
            maskIdx = torch.randperm(len(mask))[self.n_embed // 2:]
            # Random pick some never assigned loc and drop them.
            mask[maskIdx] = 1.
            dist[neverAssignedLoc] = mask
            # Update
            neverAssignedLoc = dist < self.thresh
            totalNeverAssigned = int(neverAssignedLoc.sum())
        argIdx = torch.argsort(mask, descending=True)[:(self.n_embed - totalNeverAssigned)]
        mostAssigned = _embed[argIdx]
        selectedIdx = torch.randperm(len(mostAssigned))[:totalNeverAssigned]
        _embed.data[neverAssignedLoc] = mostAssigned[selectedIdx]

        self.embed.data.copy_(_embed.transpose(0, 1))

        return