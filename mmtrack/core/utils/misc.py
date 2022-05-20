# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing as mp
import os
import platform
import warnings
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def setup_multi_processes(cfg):
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        mp.set_start_method(mp_start_method)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if ('OMP_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1):
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


def stack_batch(tensors: List[torch.Tensor],
                pad_size_divisor: int = 0,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the images to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the common height and width
    is divisible by ``pad_size_divisor``.

    Args:
        tensors (List[Tensor]): The input multiple tensors. each is a
            TCHW 4D-tensor. T denotes the number of key/reference frames.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the common height and width is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need a divisibility of 32. Defaults to 0
        pad_value (int, float): The padding value. Defaults to 0

    Returns:
       Tensor: The NTCHW 5D-tensor. N denotes the batch size.
    """
    assert isinstance(tensors, list), \
        f'Expected input type to be list, but got {type(tensors)}'
    assert len(set([tensor.ndim for tensor in tensors])) == 1, \
        f'Expected the dimensions of all tensors must be the same, ' \
        f'but got {[tensor.ndim for tensor in tensors]}'
    assert tensors[0].ndim == 4, f'Expected tensor dimension to be 4, ' \
                                 f'but got {tensors[0].ndim}'
    assert len(set([tensor.shape[0] for tensor in tensors])) == 1, \
        f'Expected the channels of all tensors must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in tensors]}'

    tensor_sizes = [(tensor.shape[-2], tensor.shape[-1]) for tensor in tensors]
    max_size = np.stack(tensor_sizes).max(0)

    if pad_size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (
            max_size +
            (pad_size_divisor - 1)) // pad_size_divisor * pad_size_divisor

    padded_samples = []
    for tensor in tensors:
        padding_size = [
            0, max_size[-1] - tensor.shape[-1], 0,
            max_size[-2] - tensor.shape[-2]
        ]
        if sum(padding_size) == 0:
            padded_samples.append(tensor)
        else:
            padded_samples.append(F.pad(tensor, padding_size, value=pad_value))

    return torch.stack(padded_samples, dim=0)
