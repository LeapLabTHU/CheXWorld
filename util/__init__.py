
import logging
import torch
from .lr_decay import param_groups_lrd, add_weight_decay
from .lr_sched import cosine_scheduler

def build_optimizer(args, model):
    if args.optim == 'sgd':
        logging.info('Use SGD Optimizer')
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        logging.info('Use AdamW Optimizer')
        if 'vit' in args.model:
            param_groups = lr_decay.param_groups_lrd(model, args.weight_decay, no_weight_decay_list=['logit_scale', 'pos_embed', 'cls_token', 'head.cls_query', 'prompt'], layer_decay=args.layer_decay)
        else:
            param_groups = lr_decay.add_weight_decay(model, args.weight_decay, skip_list=['pos_embed', 'cls_token', 'prompt'])
        
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    else:
        raise NotImplementedError

    return optimizer




def nested_to_gpu(nested_list, device):
    if isinstance(nested_list, torch.Tensor):
        return nested_list.to(device, non_blocking=True)  # Move tensor to GPU
    else:
        return [nested_to_gpu(item, device) for item in nested_list]  # Recursively process list