# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from cvpods.utils import comm
from torch.utils.data import Sampler
import numpy as np
import torch.distributed as dist
from ..registry import SAMPLERS


def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed.

    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.

    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = comm.get_rank(), comm.get_world_size()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()



@SAMPLERS.register()
class ClassAwareSampler(Sampler):
    r"""Sampler that restricts data loading to the label of the dataset.

    A class-aware sampling strategy to effectively tackle the
    non-uniform class distribution. The length of the training data is
    consistent with source data. Simple improvements based on `Relay
    Backpropagation for Effective Learning of Deep Convolutional
    Neural Networks <https://arxiv.org/abs/1512.05830>`_

    The implementation logic is referred to
    https://github.com/Sense-X/TSD/blob/master/mmdet/datasets/samplers/distributed_classaware_sampler.py

    Args:
        dataset: Dataset used for sampling.
        samples_per_gpu (int): When model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU.
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
        num_sample_class (int): The number of samples taken from each
            per-label list. Default: 1
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 num_sample_class=1):
        _rank, _num_replicas = comm.get_rank(), comm.get_world_size()
        
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        
        
        if hasattr(dataset, "datasets"):
            for d in dataset.datasets:
                assert hasattr(d, 'get_cat2imgs'), \
                        'dataset must have `get_cat2imgs` function'
        else:
            assert hasattr(dataset, 'get_cat2imgs'), \
                        'dataset must have `get_cat2imgs` function'
        assert len(dataset.datasets) ==1 , "Now not support for concat multi-datasets"
        self.dataset = dataset.datasets[0]

        if hasattr(self.dataset,'balance_pos_neg'):
            self.balance_pos_neg = self.dataset.balance_pos_neg
        else:
            self.balance_pos_neg = False

        self.num_replicas = num_replicas
        self.samples_per_gpu = samples_per_gpu
        self.rank = rank
        self.epoch = 0
        # Must be the same across all workers. If None, will use a
        # random seed shared among workers
        # (require synchronization among all workers)
        self.seed = sync_random_seed(seed)

        # Get per-label image list from dataset
        
        self.cat_dict = self.dataset.get_cat2imgs()

        if self.balance_pos_neg:
            self.num_samples = int(
                math.ceil(
                    sum([len(v) for x,v in self.cat_dict.items()]) * 1.0 / self.num_replicas /
                    self.samples_per_gpu)) * self.samples_per_gpu
        else:
            self.num_samples = int(
                math.ceil(
                    len(self.dataset) * 1.0 / self.num_replicas /
                    self.samples_per_gpu)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

        # get number of images containing each category
        self.num_cat_imgs = [len(x) for x in self.cat_dict.values()]
        # filter labels without images
        self.valid_cat_inds = [
            i for i, length in enumerate(self.num_cat_imgs) if length != 0
        ]
        self.num_classes = len(self.valid_cat_inds)
        # The number of samples taken from each per-label list
        if isinstance(num_sample_class, int):
            assert num_sample_class > 0  
        elif isinstance(num_sample_class, list):
            assert len(num_sample_class) == self.num_classes and (np.array(num_sample_class)>0).all()
        else:
            raise NotImplementedError

        self.num_sample_class = num_sample_class
            

    def __iter__(self):
        # deterministically shuffle based on epoch
        
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed) #type:ignore

        # initialize label list
        label_iter_list = RandomCycleIter(self.valid_cat_inds, generator=g)
        # initialize each per-label image list
        data_iter_dict = dict()
        for i in self.valid_cat_inds:
            data_iter_dict[i] = RandomCycleIter(self.cat_dict[i], generator=g)

        def gen_cat_img_inds(cls_list, data_dict, num_sample_cls):
            """Traverse the categories and extract `num_sample_cls` image
            indexes of the corresponding categories one by one."""
            id_indices = []
            for i in range(len(cls_list)):
                cls_idx = next(cls_list)
                if isinstance(num_sample_cls, list):
                    num_sample_cls_ = num_sample_cls[i]
                else:
                    num_sample_cls_ = num_sample_cls
                for _ in range(num_sample_cls_):
                    id = next(data_dict[cls_idx])
                    id_indices.append(id)
            return id_indices

        # deterministically shuffle based on epoch
        if isinstance(self.num_sample_class,int):
            num_bins = int(
                math.ceil(self.total_size * 1.0 / self.num_classes /
                        self.num_sample_class))
        else:
            num_bins = int(
                math.ceil(self.total_size * 1.0 / sum(self.num_sample_class)))
        indices = []
        for i in range(num_bins):
            indices += gen_cat_img_inds(label_iter_list, data_iter_dict,
                                        self.num_sample_class)

        # fix extra samples to make it evenly divisible
        if len(indices) >= self.total_size:
            indices = indices[:self.total_size]
        else:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

        # for negative random sample
        if self.balance_pos_neg:
            self.cat_dict = self.dataset.get_cat2imgs(self.epoch + self.seed)


class RandomCycleIter:
    """Shuffle the list and do it again after the list have traversed.

    The implementation logic is referred to
    https://github.com/wutong16/DistributionBalancedLoss/blob/master/mllt/datasets/loader/sampler.py

    Example:
        >>> label_list = [0, 1, 2, 4, 5]
        >>> g = torch.Generator()
        >>> g.manual_seed(0)
        >>> label_iter_list = RandomCycleIter(label_list, generator=g)
        >>> index = next(label_iter_list)
    Args:
        data (list or ndarray): The data that needs to be shuffled.
        generator: An torch.Generator object, which is used in setting the seed
            for generating random numbers.
    """  # noqa: W605

    def __init__(self, data, generator=None):
        self.data = data
        self.length = len(data)
        self.index = torch.randperm(self.length, generator=generator).numpy()
        self.i = 0
        self.generator = generator

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.i == self.length:
            self.index = torch.randperm(
                self.length, generator=self.generator).numpy()
            self.i = 0
        idx = self.data[self.index[self.i]]
        self.i += 1
        return idx
