import torch
import torch.distributed

def compute_unigrad_loss(pred, target, neg_weight=0.02):
    dense_pred = pred.reshape(-1, pred.shape[-1])
    dense_target = target.reshape(-1, target.shape[-1])

    # compute pos term
    pos_term = ((dense_pred - dense_target)**2).sum(-1).mean()

    # compute neg term
    correlation = (dense_target.T @ dense_target) / dense_target.shape[0]
    torch.distributed.all_reduce(correlation)
    correlation = correlation / torch.distributed.get_world_size()
    
    neg_term = torch.diagonal(dense_pred @ correlation @ dense_pred.T).mean()

    loss = (pos_term + neg_weight * neg_term) / pred.shape[-1]

    return loss
