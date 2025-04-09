from torch import nn
import torch
import torchvision.ops.roi_align as roi_align
import numbers
import math
from einops import rearrange

def token_to_grid(h):
    return rearrange(h, 'b (h w) c -> b c h w', h=int(h.shape[1] ** 0.5), w=int(h.shape[1] ** 0.5))

def landmark_pooling(pos, featmap, window=(3,3)):
    if isinstance(window, numbers.Number):
        window = (window, window)
    if len(featmap.shape) != 4:
        assert len(featmap.shape) == 3, 'Unknown featmap shape!'
        featmap = token_to_grid(featmap)
    
    _x1 = (window[0] // 2) * 1.0
    _x2 = (window[0] - _x1) * 1.0
    _y1 = (window[1] // 2) * 1.0
    _y2 = (window[1] - _y1) * 1.0
    B = pos.shape[0]
    N = pos.shape[1]
    h = featmap.shape[-1]
    
    ## Scale from [0,1] to [0, h]
    pos = pos * h
    pos_expanded = pos.unsqueeze(2) #[B, N, 1, 2]
    offsets = torch.tensor([[-_x1, -_y1], [_x2, _y2]], dtype=pos.dtype, device=pos.device).unsqueeze(0).unsqueeze(0) #[1,1,2,2]
    corners = pos_expanded + offsets
    boxes = corners.reshape(corners.shape[0], corners.shape[1], -1)
    boxes = [box for box in boxes]
                
    skip = roi_align(featmap, boxes, output_size = (1,1), aligned=True)
    vista = skip.view([B, N, -1])
    return vista
