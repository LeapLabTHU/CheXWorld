import logging
import functools
import os
import sys
import torch
import torch.distributed as dist
from collections import defaultdict, deque

def create_logger(args):
    logging.getLogger('PIL').setLevel(51)
    logging.getLogger("PIL.TiffImagePlugin").setLevel(51)
    logging.getLogger('matplotlib').setLevel(51)
    
    handlers = [logging.FileHandler(os.path.join(args.output_dir, f'outputs_{args.rank}.log'), mode='a+'), ]
    if args.rank == 0 and not args.no_stdout:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.DEBUG,
                        handlers=handlers)
    # print(f'logger created {args.rank}')



class MaxMeter(object):
    def __init__(self, name, fmt=':.4f', mode='max'):
        self.name = name
        self.fmt = fmt
        self.mode = mode
        assert mode in ('max', 'min')
        self.reset()
    
    def reset(self):
        self.val = -1e9 if self.mode == 'max' else 1e9
        self.ep = None
    
    def update(self, val, ep):
        if self.mode == 'max' and val > self.val:
            self.ep = ep
            self.val = val
            return True
        elif self.mode == 'min' and val < self.val:
            self.ep = ep
            self.val = val
            return True
        else:
            return False
    
    def display(self):
        fmtstr = '{mode} {name}: {val' + self.fmt + '} ({ep})'
        return fmtstr.format(**self.__dict__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, sync=False):
        self.val = val
        self.sum += val * n
        self.count += n
        if sync:
            self.synchronize_between_processes()
        self.avg = self.sum / self.count


    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'




class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)