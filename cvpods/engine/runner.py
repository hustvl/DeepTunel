#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import math
import os
import json
from collections import OrderedDict
from cvpods.structures import instances
from loguru import logger
import torch
from torch.nn.parallel import DistributedDataParallel
import time
from cvpods.checkpoint import DefaultCheckpointer
from cvpods.data import build_test_loader, build_train_loader,build_train_loader_mutiBranch,build_test_loader_mutiBranch
from cvpods.data.samplers.infinite import Infinite
from cvpods.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    inference_on_files,
    print_csv_format,
    verify_results
)
from contextlib import contextmanager, ExitStack
import numpy as np
import logging
from cvpods.utils import comm, log_every_n_seconds
from torch import nn
import datetime
from sklearn.metrics import precision_score, recall_score
import cv2
import csv
from cvpods.evaluation.registry import EVALUATOR
from cvpods.modeling.nn_utils.module_converter import maybe_convert_module
from cvpods.modeling.nn_utils.precise_bn import get_bn_modules
from cvpods.solver import build_lr_scheduler, build_optimizer
from cvpods.utils.compat_wrapper import deprecated
from cvpods.utils.dump.events import CommonMetricPrinter, JSONWriter
from sklearn.metrics import accuracy_score
from . import hooks
from .base_runner import RUNNERS, SimpleRunner

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

costom_num = dict()

def calculate_metrics(threshold, y_true, outscore, weight=1,dd=1):
    y_pred = (outscore >= threshold).astype(np.int32)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    class_counts = np.bincount(y_true)
    if len(class_counts) == 1:
        class_weights_ = None
    else:
        class_weights = class_counts / len(y_true)
        class_weights_ = np.zeros(y_true.shape)
        class_weights_[np.where(y_true==1)] = class_weights[0]
        class_weights_[np.where(y_true==0)] = class_weights[1]
    acc = accuracy_score(y_true, y_pred, sample_weight=class_weights_)
    specificity = weight*TN / (weight*TN + FP)
    sensitivity = TP / (TP + FN)
    return specificity, sensitivity, acc

def get_slide_name(image_name:str):
    ret = ''
    for x in image_name.split('_')[:-2]:
        ret += x + '_'
    return ret[:-1]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

past = dict()
def grad_check(model,params_container,print_flag=True):
    """print which layer's weight changed;

    Args:
        model:model
        params_container:(global list/dict) [] or last model's params
    Returns:
        params_container:(global list/dict) new model's params
        count: changed layers number
    """
    i = 0
    count = 0
    if isinstance(params_container,list):
        if len(params_container) !=0:
            for name, param in model.named_parameters():
                if torch.where(params_container[i]-param.clone().detach())[0].numel()!=0:
                    if print_flag:
                        print(name,':changed!')
                else:
                    if print_flag:
                        print(name,':no_changed!')
                    count += 1
                i += 1
    elif isinstance(params_container,dict):
        if len(params_container) !=0:
            for name, param in model.named_parameters():
                if torch.where(params_container[name]-param.clone().detach())[0].numel()!=0:
                    if print_flag:
                        print(name,':changed!')
                else:
                    # if print_flag:
                        # print(name,':no_changed!')
                    count += 1
    # if len(params_container) == 0:
    #     print_requires_grad_layers = True
    # else:
    #     print_requires_grad_layers = False
    # params_container = []
    # for name, param in model.named_parameters(): #查看可优化的参数有哪些
    #     if print_requires_grad_layers:
    #         if param.requires_grad:
    #             if print_flag:
    #                 print('requires grad:',name)
    #         else:
    #             if print_flag:
    #                 print('not requires grad:',name)
    #     params_container.append(param.clone().detach())
    if print_flag:
        print('*' *30)
        print('*' *30)
    return params_container,count

from pdb import set_trace

@RUNNERS.register()
class DefaultRunner(SimpleRunner):
    """
    A runner with default training logic. It does the following:

    1. Create a :class:`DefaultRunner` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`DefaultRunner` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`DefaultRunner`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in cvpods.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        runner = DefaultRunner(cfg)
        runner.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        runner.train()

    Attributes:
        scheduler:
        checkpointer (DefaultCheckpointer):
        cfg (config dict):
    """

    def __init__(self, cfg, build_model):
        """
        Args:
            cfg (config dict):
        """

        self.data_loader = self.build_train_loader(cfg)
        # Assume these objects must be constructed in this order.
        model = build_model(cfg)
        self.model = maybe_convert_module(model)
        logger.info(f"Model: \n{self.model}")

        # Assume these objects must be constructed in this order.
        self.optimizer = self.build_optimizer(cfg, self.model)

        if cfg.TRAINER.FP16.ENABLED:
            self.mixed_precision = True
            if cfg.TRAINER.FP16.TYPE == "APEX":
                from apex import amp
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level=cfg.TRAINER.FP16.OPTS.OPT_LEVEL
                )
        else:
            self.mixed_precision = False

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            torch.cuda.set_device(comm.get_local_rank())
            if cfg.MODEL.DDP_BACKEND == "torch":
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=False,
                    find_unused_parameters=False
                )
            elif cfg.MODEL.DDP_BACKEND == "apex":
                from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel
                self.model = ApexDistributedDataParallel(self.model)
            else:
                raise ValueError("non-supported DDP backend: {}".format(cfg.MODEL.DDP_BACKEND))

        super().__init__(
            self.model,
            self.data_loader,
            self.optimizer,
        )

        if not cfg.SOLVER.LR_SCHEDULER.get("EPOCH_WISE", False):
            epoch_iters = -1
        else:
            epoch_iters = cfg.SOLVER.LR_SCHEDULER.get("EPOCH_ITERS")
            logger.warning(f"Setup LR Scheduler in EPOCH mode: {epoch_iters}")

        auto_scale_config(cfg, self.data_loader)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, epoch_iters=epoch_iters)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DefaultCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,

            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.start_iter = 0
        self.start_epoch = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
        self.window_size = cfg.TRAINER.WINDOW_SIZE

        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume = resume
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
        if self.max_epoch is not None:
            if isinstance(self.data_loader.sampler, Infinite):
                length = len(self.data_loader.sampler.sampler)
            else:
                length = len(self.data_loader)
            self.start_epoch = self.start_iter // length

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg
        # cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.OptimizationHook(
                accumulate_grad_steps=cfg.SOLVER.BATCH_SUBDIVISIONS,
                grad_clipper=None,
                mixed_precision=cfg.TRAINER.FP16.ENABLED
            ),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer,
                cfg.SOLVER.CHECKPOINT_PERIOD,
                max_iter=self.max_iter,
                max_epoch=self.max_epoch
            ))

        def test_and_save_results():
            evaluation_type = self.cfg.TEST.get('EVALUATION_TYPE',None)

            if evaluation_type is not None:
                evaluator = EVALUATOR.get(evaluation_type)()
            else:
                evaluator = None
            self._last_eval_results = self.test(self.cfg, self.model,evaluators=evaluator)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(
                self.build_writers(), period=self.cfg.GLOBAL.LOG_INTERVAL
            ))
            # Put `PeriodicDumpLog` after writers so that can dump all the files,
            # including the files generated by writers

        return ret

    def build_writers(self):
        """
        Build a list of :class:`EventWriter` to be used.
        It now consists of a :class:`CommonMetricPrinter`,
        :class:`TensorboardXWriter` and :class:`JSONWriter`.

        Args:
            output_dir: directory to store JSON metrics and tensorboard events
            max_iter: the total number of iterations

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(
                self.max_iter,
                window_size=self.window_size,
                epoch=self.max_epoch,
            ),
            JSONWriter(
                os.path.join(self.cfg.OUTPUT_DIR, "metrics.json"),
                window_size=self.window_size
            )
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        if self.max_epoch is None:
            logger.info("Starting training from iteration {}".format(self.start_iter))
        else:
            logger.info("Starting training from epoch {}".format(self.start_epoch))

        super().train(self.start_iter, self.start_epoch, self.max_iter)

        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`cvpods.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, **kwargs):
        """
        It now calls :func:`cvpods.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer, **kwargs)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`cvpods.data.build_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`cvpods.data.build_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_test_loader(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError(
            # TODO: add this tutorial
            """
If you want DefaultRunner to automatically run evaluation,
please implement `build_evaluator()` in subclasses (see train_net.py for example).
Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
            """
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None, output_folder=None):
        """
        Args:
            cfg (config dict):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            # assert evaluators is not None
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    from cvpods.evaluation import build_evaluator
                    evaluator = build_evaluator(
                        cfg, dataset_name, data_loader.dataset, output_folder=output_folder)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultRunner.test(evaluators=)`, "
                        "or implement its `build_evaluator` method.")
                    results[dataset_name] = {}
                    continue

            results_i = inference_on_dataset(model, data_loader, evaluator)
            # if cfg.TEST.ON_FILES:
            #     results_i = inference_on_files(evaluator)
            # else:
            #     results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                try:
                    print_csv_format(results_i)
                except:
                    f1score = np.array(results_i['f1score'])
                    f1score[np.isnan(f1score)] = 0
                    max_f1score, argmax_f1score = f1score.max(), f1score.argmax()
                    print_results = {"precision:":results_i['precision'][argmax_f1score],"recall:":results_i['recall'][argmax_f1score],'f1score':max_f1score}
                    logger.info(print_results)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def auto_scale_config(cfg, dataloader):
    max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
    max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER

    subdivision = cfg.SOLVER.BATCH_SUBDIVISIONS
    # adjust lr by batch_subdivisions
    cfg.SOLVER.OPTIMIZER.BASE_LR *= subdivision

    """
    Here we use batch size * subdivision to simulator large batch training
    """
    if max_epoch:
        epoch_iter = math.ceil(
            len(dataloader.dataset) / (cfg.SOLVER.IMS_PER_BATCH * subdivision))

        if max_iter is not None:
            logger.warning(
                f"Training in EPOCH mode, automatically convert {max_epoch} epochs "
                f"into {max_epoch*epoch_iter} iters...")

        cfg.SOLVER.LR_SCHEDULER.MAX_ITER = max_epoch * epoch_iter
        cfg.SOLVER.LR_SCHEDULER.STEPS = [
            x * epoch_iter for x in cfg.SOLVER.LR_SCHEDULER.STEPS
        ]
        cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS = int(
            cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS * epoch_iter)
        cfg.SOLVER.CHECKPOINT_PERIOD = epoch_iter * cfg.SOLVER.CHECKPOINT_PERIOD
        cfg.TEST.EVAL_PERIOD = epoch_iter * cfg.TEST.EVAL_PERIOD
    else:
        epoch_iter = -1

    cfg.SOLVER.LR_SCHEDULER.EPOCH_ITERS = epoch_iter


@RUNNERS.register()
@deprecated("Use DefaultRunner instead.")
class DefaultTrainer(DefaultRunner):
    pass


@RUNNERS.register()
class MultiBranchRunner(SimpleRunner):

    def __init__(self, cfg, build_model):
        """
        Args:
            cfg (config dict):
        """
        self.data_loader = build_train_loader_mutiBranch(cfg)
        if cfg.MODEL.FCOS.get('WITH_TEACHER',None):
            model, self.teacher_model = build_model(cfg)
        else:
            model= build_model(cfg)
            self.teacher_model = None

        self.model = maybe_convert_module(model)
        if self.teacher_model is not None:
            self.teacher_model = maybe_convert_module(self.teacher_model)
        logger.info(f"Model: \n{self.model}")

        self.optimizer = build_optimizer(cfg, self.model)

        if cfg.TRAINER.FP16.ENABLED:
            self.mixed_precision = True
            if cfg.TRAINER.FP16.TYPE == "APEX":
                from apex import amp
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level=cfg.TRAINER.FP16.OPTS.OPT_LEVEL
                )

        else:
            self.mixed_precision = False
        

        if comm.get_world_size() > 1:
            torch.cuda.set_device(comm.get_local_rank())
            if cfg.MODEL.DDP_BACKEND == "torch":
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=False,
                    find_unused_parameters=False
                )

            elif cfg.MODEL.DDP_BACKEND == "apex":
                from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel
                self.model = ApexDistributedDataParallel(self.model)
            else:
                raise ValueError("non-supported DDP backend: {}".format(cfg.MODEL.DDP_BACKEND))


        super().__init__(
            self.model,
            self.data_loader,
            self.optimizer,
        )

        if not cfg.SOLVER.LR_SCHEDULER.get("EPOCH_WISE", False):
            epoch_iters = -1
        else:
            epoch_iters = cfg.SOLVER.LR_SCHEDULER.get("EPOCH_ITERS")
            logger.warning(f"Setup LR Scheduler in EPOCH mode: {epoch_iters}")

        auto_scale_config(cfg, self.data_loader)
        self.scheduler = build_lr_scheduler(cfg, self.optimizer, epoch_iters=epoch_iters)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DefaultCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            "model_iter_",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        if self.teacher_model is not None:
            self.teacher_checkpointer = DefaultCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                self.teacher_model,
                cfg.OUTPUT_DIR,
                "teacher_model_iter_",
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
        else:
            self.teacher_checkpointer = None
        self.start_iter = 0
        self.start_epoch = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
        self.window_size = cfg.TRAINER.WINDOW_SIZE

        self.cfg = cfg


        self.register_hooks(self.build_hooks())        


    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """

        self.checkpointer.resume = resume
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
        # _ = (self.teacher_checkpointer.resume_or_load(
        #     self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)

        if self.teacher_checkpointer:

            weight = self.cfg.MODEL.get('TEACHER_WEIGHTS',self.cfg.MODEL.WEIGHTS)
            # aa = torch.load(weight)
            self.teacher_checkpointer.resume_or_load(weight,resume=resume)
        if self.max_epoch is not None:
            if isinstance(self.data_loader.sampler, Infinite):
                length = len(self.data_loader.sampler.sampler)
            else:
                length = len(self.data_loader)
            self.start_epoch = self.start_iter // length
        
        
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg
        # cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.OptimizationHook(
                accumulate_grad_steps=cfg.SOLVER.BATCH_SUBDIVISIONS,
                grad_clipper=None,
                mixed_precision=cfg.TRAINER.FP16.ENABLED
            ),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer,
                cfg.SOLVER.CHECKPOINT_PERIOD,
                max_iter=self.max_iter,
                max_epoch=self.max_epoch
            ))
            if self.teacher_checkpointer is not None:
                ret.append(hooks.PeriodicCheckpointer(
                    self.teacher_checkpointer,
                    cfg.SOLVER.CHECKPOINT_PERIOD,
                    max_iter=self.max_iter,
                    max_epoch=self.max_epoch
                ))
        def test_and_save_results():
            evaluation_type = self.cfg.TEST.get('EVALUATION_TYPE',None)

            if evaluation_type is not None:
                evaluator = EVALUATOR.get(evaluation_type)()
            else:
                evaluator = None
            self._last_eval_results = self.test(self.cfg, self.model,evaluators=evaluator)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(
                self.build_writers(), period=100
            ))
            # Put `PeriodicDumpLog` after writers so that can dump all the files,
            # including the files generated by writers

        return ret


    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        if self.max_epoch is None:
            logger.info("Starting training from iteration {}".format(self.start_iter))
        else:
            logger.info("Starting training from epoch {}".format(self.start_epoch))

        from cvpods.utils.dump.events import EventStorage
        avg_meters = {}
        pbar = tqdm(total=self.max_iter)   

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.iter = self.start_iter
                self.before_train()

                self.start_training = True
                for self.iter in range(self.start_iter, self.max_iter):
                    self.inner_iter = 0
                    self.before_step()
                    # by default, a step contains data_loading and model forward,
                    # loss backward is executed in after_step for better expansibility
                    self.run_step()

                    self.after_step()

                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1

            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()


        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
      
        assert self.model.training, "[IterRunner] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self.epoch += 1
            costom_num['found_num'] = costom_num['all_num'] \
                    = costom_num['wrong_loc_num']= costom_num['ori_num'] \
                    = costom_num['wrong_cls_num'] = 0
                    
            if hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(self.epoch)
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)

        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        data[0]['iter'] = self.iter
        loss_dict = self.model(data,self.teacher_model)


        self.inner_iter += 1


        losses = sum([
            metrics_value for metrics_value in loss_dict.values()
            if metrics_value.requires_grad
        ])
        self._detect_anomaly(losses, loss_dict)

        self._write_metrics(loss_dict, data_time)

        self.step_outputs = {
            "loss_for_backward": losses,
        }

        
    

    @classmethod
    def test(cls, cfg, model, evaluators=None, output_folder=None,teacher_model=None):
        """
        Args:
            cfg (config dict):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()

        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            # if teacher_model is not None:
                # data_loader = build_test_loader_mutiBranch(cfg)
            # else:
            data_loader = build_test_loader(cfg)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    from cvpods.evaluation import build_evaluator
                    evaluator = build_evaluator(
                        cfg, dataset_name, data_loader.dataset, output_folder=output_folder)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultRunner.test(evaluators=)`, "
                        "or implement its `build_evaluator` method.")
                    results[dataset_name] = {}
                    continue

            # results_i = inference_on_dataset(model, data_loader, evaluator)
            if cfg.TEST.ON_FILES:
                results_i = inference_on_files(evaluator)
            else:
                results_i = inference_on_dataset(model, data_loader, evaluator,teacher_model=teacher_model)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                try:
                    print_csv_format(results_i)
                except:
                    f1score = np.array(results_i['f1score'])
                    f1score[np.isnan(f1score)] = 0
                    max_f1score, argmax_f1score = f1score.max(), f1score.argmax()
                    print_results = {"precision:":results_i['precision'][argmax_f1score],"recall:":results_i['recall'][argmax_f1score],'f1score':max_f1score}
                    logger.info(print_results)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def build_writers(self):
        """
        Build a list of :class:`EventWriter` to be used.
        It now consists of a :class:`CommonMetricPrinter`,
        :class:`TensorboardXWriter` and :class:`JSONWriter`.

        Args:
            output_dir: directory to store JSON metrics and tensorboard events
            max_iter: the total number of iterations

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(
                self.max_iter,
                window_size=self.window_size,
                epoch=self.max_epoch,
            ),
            JSONWriter(
                os.path.join(self.cfg.OUTPUT_DIR, "metrics.json"),
                window_size=self.window_size
            )
        ]


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def draw(inputs,outputs=None):
    from pdb import set_trace
    set_trace()
    if outputs is None:
        outputs = [None for _ in inputs]
    for inp, output in zip(inputs, outputs):
        img = np.array(inputs[0]['image'].permute(1,2,0))
        img = img.copy()
        img_c = img.copy()
        anns = inp['annotations']
        file_name = os.path.basename(inputs[0]['file_name'])
        if output is None:
            output = [None for _ in anns]
        for ann,o_l in zip(anns, output):
            bbox = ann['bbox']
            label = ann['category_id']
            
            img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0) if label==0 else (0,255,0),1)
            if o_l is not None:
                img_c = cv2.rectangle(img_c,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0) if o_l==0 else (0,255,0),1)

        if output is not None:
             img = np.hstack([img,img_c])
        cv2.imwrite('/data/taoranyi/temp_for_wulianjun/MixTeacher/vis/'+file_name,img)

@RUNNERS.register()
class DiaoWangRunner(SimpleRunner):

    def __init__(self, cfg, build_model):
        """
        Args:
            cfg (config dict):
        """
        if not cfg.TRAINER.get('MULTI_BRANCH',False):
            self.data_loader = build_train_loader(cfg)
        else:
            self.data_loader = build_train_loader_mutiBranch(cfg)

        if cfg.MODEL.get('WITH_TEACHER',None):
            model,self.teacher_model = build_model(cfg)
        else:
            model= build_model(cfg)
            self.teacher_model = None
        self.model = maybe_convert_module(model)

        if self.teacher_model is not None:
            self.teacher_model = maybe_convert_module(self.teacher_model)
        logger.info(f"Model: \n{self.model}")
        self.optimizer = build_optimizer(cfg, self.model)

        if cfg.TRAINER.FP16.ENABLED:
            self.mixed_precision = True
            if cfg.TRAINER.FP16.TYPE == "APEX":
                from apex import amp
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level=cfg.TRAINER.FP16.OPTS.OPT_LEVEL
                )

        else:
            self.mixed_precision = False
        

        if comm.get_world_size() > 1:
            torch.cuda.set_device(comm.get_local_rank())
            if cfg.MODEL.DDP_BACKEND == "torch":
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=False,
                    find_unused_parameters=False
                )

            elif cfg.MODEL.DDP_BACKEND == "apex":
                from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel
                self.model = ApexDistributedDataParallel(self.model)
            else:
                raise ValueError("non-supported DDP backend: {}".format(cfg.MODEL.DDP_BACKEND))


        super().__init__(
            self.model,
            self.data_loader,
            self.optimizer,
        )

        if not cfg.SOLVER.LR_SCHEDULER.get("EPOCH_WISE", False):
            epoch_iters = -1
        else:
            epoch_iters = cfg.SOLVER.LR_SCHEDULER.get("EPOCH_ITERS")
            logger.warning(f"Setup LR Scheduler in EPOCH mode: {epoch_iters}")

        auto_scale_config(cfg, self.data_loader)
        self.scheduler = build_lr_scheduler(cfg, self.optimizer, epoch_iters=epoch_iters)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DefaultCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            "model_iter_",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        if self.teacher_model is not None:
            self.teacher_checkpointer = DefaultCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                self.teacher_model,
                cfg.OUTPUT_DIR,
                "teacher_model_iter_",
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
        else:
            self.teacher_checkpointer = None
        self.start_iter = 0
        self.start_epoch = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
        self.window_size = cfg.TRAINER.WINDOW_SIZE
        self.fliter_thre = cfg.TRAINER.get('fliter_thre',0)
        self.cfg = cfg


        self.register_hooks(self.build_hooks())        


    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """

        self.checkpointer.resume = resume
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
        # _ = (self.teacher_checkpointer.resume_or_load(
        #     self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)

        if self.teacher_checkpointer:

            weight = self.cfg.MODEL.get('TEACHER_WEIGHTS',self.cfg.MODEL.WEIGHTS)
            # aa = torch.load(weight)
            self.teacher_checkpointer.resume_or_load(weight,resume=resume)
        if self.max_epoch is not None:
            if isinstance(self.data_loader.sampler, Infinite):
                length = len(self.data_loader.sampler.sampler)
            else:
                length = len(self.data_loader)
            self.start_epoch = self.start_iter // length


    @classmethod
    def test(cls, cfg, model, output_folder=None,evaluators=None):
        results = OrderedDict()

        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            print("dataset_name:",dataset_name)
            if idx>0:
                continue
            data_loader = cls.build_test_loader(cfg)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            # assert evaluators is not None

            num_devices = comm.get_world_size()
            logger = logging.getLogger(__name__)
            
            logger.info("Start inference on {} batches".format(len(data_loader)))

            total = len(data_loader)  # inference data loader must have a fixed length
        

            num_warmup = min(5, total - 1)
            start_time = time.perf_counter()
            total_data_time = 0
            total_compute_time = 0
            total_eval_time = 0
            outputs_all = {'0.5':[]}
            score_thres = [k for k,_ in outputs_all.items()]
            inputs_all = []
            inputs_info = []
            outputs_dict_all = dict()
            outputs_scores = []
            # outputs_like_inputs = dict()
            ids = 0
            # all_high_qlt_patches = []
            # all_low_qlt_patches = []
            # fliter_score = 0.8
            lengths = []
            ret = dict()
            f_txt = open('/data/taoranyi/temp_for_wulianjun/time_cost_320.txt','w')
            f_txt2 = open('/data/taoranyi/temp_for_wulianjun/aaaaa.txt','w')
            with ExitStack() as stack:
                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())

                start_data_time = time.perf_counter()
                for idx, inputs in enumerate(data_loader):
                    total_data_time += time.perf_counter() - start_data_time

                    if idx == num_warmup:
                        start_time = time.perf_counter()
                        total_data_time = 0
                        total_compute_time = 0
                        total_eval_time = 0
                    # if idx % 50 ==0:
                    #     logger.info(str(idx))
                    start_compute_time = time.perf_counter()
                    # each_image_time_start = time.perf_counter()
                    outputs = model(inputs)

                    # inp_each = inputs[0]['instances'].gt_classes.numpy()
                    # outscore_each = np.array(outputs.squeeze(-1).cpu())
                    # y_pred_each = (outscore_each >= 0.5).astype(np.int32)
                    # aaa,bbb,ccc = inp_each.sum(), y_pred_each.sum(), np.sum((inp_each == 1) & (y_pred_each == 1))
                    # recall_ = recall_score(inp_each, y_pred_each, average='binary')
                    # precision_ = precision_score(inp_each, y_pred_each, average='binary')
                    # f1_scores_ = 2*recall_*precision_ / (recall_+precision_+1e-6)
                    # specificity, sensitivity, acc = calculate_metrics(0.5,inp_each,outscore_each)
                    # names = os.path.basename(inputs[0]['file_name'])
                    # f_txt2.write(f'{names} {recall_},{precision_},{f1_scores_}')
                    # f_txt2.write('\n')
                    # each_image_time_end = time.perf_counter()
                    # each_image_time = round((each_image_time_end-each_image_time_start)*1000,1)
                    # f_txt.write(f'{each_image_time}')
                    # f_txt.write('\n')
                    # slide_name = get_slide_name(os.path.basename(inputs[0]['file_name']))
                    # if  slide_name not in outputs_like_inputs:
                    #     outputs_like_inputs[slide_name] = dict()

                    # outputs_like_inputs[slide_name][os.path.basename(inputs[0]['file_name']).strip('.png')] = \
                    #     {'bboxes':inputs[0]['instances'].gt_boxes.tensor.int().numpy().tolist(),
                    #      'labels':(outputs>0.5).int().reshape(-1).tolist()}


                    outputs_scores.extend(outputs.reshape(-1).tolist()) 


                    if cfg.DATASETS.get('SHOW', False):
                        draw(inputs,[outputs])

                    input_list = inputs[0]["instances"].gt_classes.tolist()
                    inputs_all.extend(input_list)
                    inputs_info.extend([{'id':ids + i,'species':inputs[0]["species"],'slide_name':get_slide_name(os.path.basename(inputs[0]["file_name"]))} for i in range(
                                       len(input_list))])
                    
                    # f1_scores_max = 0
                    # recall_max = 0
                    # precision_max = 0
                    # add_ = False
                    for thre in score_thres:
                        outputs = outputs.reshape(-1,1)
                        outputs_thre = (outputs > float(thre)).int().squeeze(-1).cpu().tolist()
                        outputs_all[thre].extend(outputs_thre)

                    ids += len(input_list)
                    patch_name = os.path.basename(inputs[0]['file_name']).strip('png')
                    lengths.append({patch_name:len(input_list)})
                    ins = inputs[0]['instances']
                    ins.gt_classes =  (outputs > 0.5).int().squeeze(-1).cpu()
                    
                    slide_name = get_slide_name(patch_name)
                    if slide_name not in outputs_dict_all:
                        outputs_dict_all[slide_name] = dict()
                    outputs_dict_all[slide_name][patch_name.strip('.png')] = {'bboxes':inputs[0]['instances'].gt_boxes.tensor.int().tolist(),
                                            'labels':ins.gt_classes.tolist()}

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    total_compute_time += time.perf_counter() - start_compute_time

                    start_eval_time = time.perf_counter()
                    
                    # evaluator.process(inputs, outputs)
                    total_eval_time += time.perf_counter() - start_eval_time

                    iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                    data_seconds_per_iter = total_data_time / iters_after_start
                    compute_seconds_per_iter = total_compute_time / iters_after_start
                    eval_seconds_per_iter = total_eval_time / iters_after_start
                    total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                    if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                        eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                        log_every_n_seconds(
                            logging.INFO,
                            (
                                f"Inference done {idx + 1}/{total}. "
                                f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                f"ETA={eta}"
                            ),
                            n=1,
                        )
                    start_data_time = time.perf_counter()

            # Measure the time only for this worker (before the synchronization barrier)
            total_time = time.perf_counter() - start_time
            total_time_str = str(datetime.timedelta(seconds=total_time))
            # NOTE this format is parsed by grep
            logger.info(
                "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                    total_time_str, total_time / (total - num_warmup), num_devices
                )
            )
            total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
            logger.info(
                "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                    total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
                )
            )
            results_i = dict()
            for thre,out_list in outputs_all.items():
                recall = recall_score(inputs_all, out_list, average='binary')
                precision = precision_score(inputs_all, out_list, average='binary')
                acc = (np.array(out_list)==np.array(inputs_all)).sum() / float(len(inputs_all))
                results_i[thre] = {"precision":precision,"recall":recall,"acc":acc}

            print(results_i)
            f1_scores = np.array([2*v['precision'] *v['recall']/(v['precision'] + v['recall']+1e-6)  for thre, v in results_i.items()])
            max_arg = np.argmax(f1_scores)
            results_i = results_i[score_thres[max_arg]]
            results[dataset_name] = results_i
            # out = outputs_all[score_thre]

            def get_split_results(out,inputs_all,inputs_info):
                all_slides = list(set([x['slide_name'] for x in inputs_info]))
                infos = ['LLC','Liu','Gan','Rabbit']
                results_split = dict()
                for info in infos:
                    aa = list(filter(lambda x: x['species'] == info or info in x['slide_name'],inputs_info))
                    indices = np.array([x['id'] for x in aa],dtype=np.int64)
                    inp = np.array(inputs_all)[indices].tolist()
                    max_f1_scores = 0
                    max_results = dict()
                    for thre,out_list in outputs_all.items():
                        out_ = np.array(out_list)[indices].tolist()
                        recall_ = recall_score(inp, out_, average='binary')
                        precision_ = precision_score(inp, out_, average='binary')
                        acc_ = (np.array(out_)==np.array(inp)).sum() / float(len(inp))
                        f1_scores_ = 2*recall_*precision_ / (recall_+precision_+1e-6)

                        if f1_scores_>=max_f1_scores:
                            max_f1_scores = f1_scores_
                            max_results = {"precision":precision_,"recall":recall_,"acc":acc_,'f1_scores':max_f1_scores,'score_thre':thre,
                            'GP':sum(inp),'GN':len(inp) - sum(inp),'PP':sum(out_),'PN':len(out_) - sum(out_)}

                    results_split[info] = {"precision":'{:.3f}'.format(max_results['precision']),"recall":'{:.3f}'.format(max_results['recall']),
                    "acc":'{:.3f}'.format(max_results['acc']),'f1_scores':'{:.3f}'.format(max_results['f1_scores']),'score_thre':max_results['score_thre'],
                    'GP':max_results['GP'],'GN':max_results['GN'],'PP':max_results['PP'],'PN':max_results['PN']
                    }
                print(results_split)

                # def get_statis_tabel(results_split):
                #     # every_slide_statis['sum'] = [sum(v[0] for _,v in every_slide_statis.items()),sum(v[1] for _,v in every_slide_statis.items()),sum(v[2] for _,v in every_slide_statis.items()) ]
                #     # mean_statis = 
                #     with open(os.path.join(output_folder+'/results_split.csv'),'w') as f:
                #         writer = csv.writer(f)
                #         writer.writerow(['name','precision','recall','acc','f1_scores','score_thre','GP','GN','PP','PN',])
                #         for info,val in results_split.items():
                #             prit = [(info,val['precision'],val['recall'],val['acc'],val['f1_scores'],val['score_thre'],val['GP'],val['GN'],val['PP'],val['PN'])]
                #             writer.writerows(prit)
                #         # mean
                # get_statis_tabel(results_split)
            
            get_split_results(outputs_all,inputs_all,inputs_info)

            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                try:
                    print_csv_format(results_i)
                except:
                    print_results = "precision:{},recall:{},acc:{},f1score:{}".format(results_i['precision'],results_i['recall'],acc,f1_scores[max_arg])
                    log_every_n_seconds(logging.INFO,print_results,n=1)

            # from pdb import set_trace
            # set_trace()
            ret['inputs_all'] = inputs_all
            # ret['outputs_all'] = outputs_all
            # ret['outputs_all'] = outputs_scores
            
            ret['length'] = lengths
            name = dataset_name
            # with open(output_folder+'/result_{}_scores.json'.format(name),'w') as f:
            #     json.dump(ret,f)

            ret['outputs_all'] = outputs_all
            with open(output_folder+'/result_{}.json'.format(name),'w') as f:
                json.dump(ret,f)

            # with open(output_folder+'/result_{}_like_inputs.json'.format(name),'w') as f:
            #     json.dump(outputs_like_inputs,f)
            
            def show_auc(y_true,y_scores,xlim_min=0,xlim_max=1,ylim_min=0,ylim_max=1.05,):
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)

                # 计算AUC
                auc_value = auc(fpr, tpr)

                # 绘制AUC曲线
                fig = plt.figure()
                plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_value)
                plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
                plt.xlim([xlim_min, xlim_max])
                plt.ylim([ylim_min, ylim_max])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend(loc="lower right")
                fig.savefig('/data/taoranyi/temp_for_wulianjun/auc_all.png')

            def get_metric(inputs_all,outputs_scores,inputs_info, dd=1):
                infos = ['LLC','Liu','Gan','Rabbit']
                colors = ['darkorange','blue','red','green']
                inp = np.array(inputs_all)
                outscore = np.array(outputs_scores)

                def calculate_metrics(threshold, y_true, outscore, weight,dd=1):
                    y_pred = (outscore >= threshold).astype(np.int32)
                    TP = np.sum((y_true == 1) & (y_pred == 1))
                    FP = np.sum((y_true == 0) & (y_pred == 1))
                    TN = np.sum((y_true == 0) & (y_pred == 0))
                    FN = np.sum((y_true == 1) & (y_pred == 0))
                    class_counts = np.bincount(y_true)
                    class_weights = class_counts / len(y_true)
                    class_weights_ = np.zeros(y_true.shape)
                    class_weights_[np.where(y_true==1)] = class_weights[0]
                    class_weights_[np.where(y_true==0)] = class_weights[1]
                    acc = accuracy_score(y_true, y_pred, sample_weight=class_weights_)
                    specificity = weight*TN / (weight*TN + FP)
                    sensitivity = TP / (TP + FN)
                    return specificity, sensitivity, acc
                # from pdb import set_trace
                # set_trace()
                specificity, sensitivity, acc = calculate_metrics(0.5, inp, outscore, sum(inp)/len(inp), dd)
                print(f"all: specificity:{specificity},sensitivity:{sensitivity},acc:{acc}" )
                
                # for i,(info,color) in enumerate(zip(infos,colors)):
                #     aa = list(filter(lambda x: x['species'] == info or info in x['slide_name'],inputs_info))
                #     indices = np.array([x['id'] for x in aa],dtype=np.int64)
                #     inp_i = inp[indices].tolist()
                #     outscore = np.array(outputs_scores)[indices].tolist()
                #     y_true = np.array(inp_i)
                #     outscore = np.array(outscore)
                #     weight = sum(y_true)/(len(y_true) + 1e-8)
                    
                #     specificity, sensitivity, acc = calculate_metrics(0.5, y_true, outscore, weight, dd)
                #     print(f"{info}: specificity:{specificity},sensitivity:{sensitivity},acc:{acc}" )


            def show_split_auc(inputs_all,outputs_scores,inputs_info,xlim_min=0,xlim_max=1,ylim_min=0,ylim_max=1.05,):
                infos = ['LLC','Liu','Gan','Rabbit']
                colors = ['darkorange','blue','red','green']
                with open('/data3/scripts/human_results.json') as f:
                    human_results = json.load(f)

                fig = plt.figure()
                inp = np.array(inputs_all).tolist()
                outscore = np.array(outputs_scores).tolist()
                fpr1, tpr1, thresholds1 = roc_curve(inp, outscore)
                specificitys = []
                sensitivitys = []

                y_true = np.array(inp)
                outscore = np.array(outscore)
                from joblib import Parallel, delayed
                weight = sum(y_true)/len(y_true)

                def calculate_metrics(threshold, y_true, outscore, weight):
                    y_pred = (outscore >= threshold).astype(np.int32)
                    TP = np.sum((y_true == 1) & (y_pred == 1))
                    FP = np.sum((y_true == 0) & (y_pred == 1))
                    TN = np.sum((y_true == 0) & (y_pred == 0))
                    FN = np.sum((y_true == 1) & (y_pred == 0))
                    specificity = weight*TN / (weight*TN + FP)
                    sensitivity = TP / (TP + FN)
                    return specificity, sensitivity
                
                thresholds = thresholds1[1:]
                num_thresholds = len(thresholds)
                specificities = np.zeros(num_thresholds)
                sensitivities = np.zeros(num_thresholds)
                num_cores = 2  # Number of CPU cores to use
                results = Parallel(n_jobs=num_cores)(
                    delayed(calculate_metrics)(threshold, y_true, outscore, weight)
                    for threshold in tqdm(thresholds)
                )
                specificities, sensitivities = zip(*results)
                specificitys = np.array(specificities)
                sensitivitys = np.array(sensitivities)
                fig = plt.figure()
                roc_auc1 = auc(1-np.array(specificitys), tpr1[1:])
                plt.plot(1-np.array(specificitys), tpr1[1:], color='black', label='Weighted ROC curve (AUC = {:.2f})'.format(roc_auc1))
                plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
                plt.xlim([xlim_min, xlim_max])
                plt.ylim([ylim_min, ylim_max])
                plt.xlabel('1-Specificity (%)')
                plt.ylabel('Sensitivity (%)')
                plt.title('Weighted Receiver Operating Characteristic')
                plt.scatter(1-0.8011, 1.0, facecolors='none', edgecolors='black', linewidths=1, s=100, label='SoftWare ALL')
                plt.text(1-0.8011, 1.0, 'S', color='black', fontsize=8, ha='center', va='center')
                # for idx, (human_id, v) in enumerate(human_results.items()):
                    # plt.scatter(1-v['all']['Specificity'], v['all']['Sensitivity'], facecolors='none', edgecolors='black', linewidths=1, s=100, label=None if idx>0 else 'Expert 1~6')
                    # plt.text(1-v['all']['Specificity'], v['all']['Sensitivity'], str(idx+1), color='black', fontsize=8, ha='center', va='center')

                plt.legend(loc="lower right")
                fig.savefig('/data/taoranyi/temp_for_wulianjun/auc_software_all.svg')

                for i,(info,color) in enumerate(zip(infos,colors)):
                    aa = list(filter(lambda x: x['species'] == info or info in x['slide_name'],inputs_info))
                    indices = np.array([x['id'] for x in aa],dtype=np.int64)
                    inp = np.array(inputs_all)[indices].tolist()
                    outscore = np.array(outputs_scores)[indices].tolist()
                    fpr1, tpr1, thresholds1 = roc_curve(inp, outscore)
                    specificitys = []
                    sensitivitys = []
                    y_true = np.array(inp)
                    outscore = np.array(outscore)
                    weight = sum(y_true)/len(y_true)
                    thresholds = thresholds1[1:]
                    num_thresholds = len(thresholds)
                    specificities = np.zeros(num_thresholds)
                    sensitivities = np.zeros(num_thresholds)
                    num_cores = 2  # Number of CPU cores to use
                    results = Parallel(n_jobs=num_cores)(
                        delayed(calculate_metrics)(threshold, y_true, outscore, weight)
                        for threshold in tqdm(thresholds)
                    )
                    specificities, sensitivities = zip(*results)
                    specificitys = np.array(specificities)
                    sensitivitys = np.array(sensitivities)
                    fig = plt.figure()
                    roc_auc1 = auc(1-np.array(specificitys), tpr1[1:])
                    plt.plot(1-np.array(specificitys), tpr1[1:], color=color, label='Weighted ROC curve {} (AUC = {:.2f})'.format(infos[i],roc_auc1))
                    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
                    plt.xlim([xlim_min, xlim_max])
                    plt.ylim([ylim_min, ylim_max])
                    plt.xlabel('1-Specificity (%)')
                    plt.ylabel('Sensitivity (%)')
                    plt.title(f'Weighted Receiver Operating Characteristic of {info}')
                    if i==0:
                        plt.scatter(1-0.7656, 1.0, facecolors='none', edgecolors='darkorange', linewidths=1, s=100, label='SoftWare LLC')
                        plt.text(1-0.7656, 1.0, 'S', color='darkorange', fontsize=8, ha='center', va='center')
                        
                        # for idx, (human_id, v) in enumerate(human_results.items()):
                            # plt.scatter(1-v['LLC']['Specificity'], v['LLC']['Sensitivity'], facecolors='none', edgecolors='darkorange', linewidths=1, s=100,label=None if idx>0 else 'Experts 1~6 LLC')
                            # plt.text(1-v['LLC']['Specificity'], v['LLC']['Sensitivity'], str(idx+1), color='darkorange', fontsize=8, ha='center', va='center')
                    if i==1:
                        plt.scatter(1-0.7621, 1.0, facecolors='none', edgecolors='blue', linewidths=1, s=100, label='SoftWare Liu')
                        plt.text(1-0.7621, 1.0, 'S', color='blue', fontsize=8, ha='center', va='center')
                        # for idx, (human_id, v) in enumerate(human_results.items()):
                            # plt.scatter(1-v['Liu']['Specificity'], v['Liu']['Sensitivity'], facecolors='none', edgecolors='blue', linewidths=1, s=100,label=None if idx>0 else 'Experts 1~6 Liu')
                            # plt.text(1-v['Liu']['Specificity'], v['Liu']['Sensitivity'], str(idx+1), color='blue', fontsize=8, ha='center', va='center')
                    if i==2:
                        plt.scatter(1-0.7017, 1.0, facecolors='none', edgecolors='red', linewidths=1, s=100, label='SoftWare Gan')
                        plt.text(1-0.7017, 1.0, 'S', color='red', fontsize=8, ha='center', va='center')
                        # for idx, (human_id, v) in enumerate(human_results.items()):
                            # plt.scatter(1-v['Gan']['Specificity'], v['Gan']['Sensitivity'], facecolors='none', edgecolors='red', linewidths=1, s=100,label=None if idx>0 else 'Experts 1~6 Gan')
                            # plt.text(1-v['Gan']['Specificity'], v['Gan']['Sensitivity'], str(idx+1), color='red', fontsize=8, ha='center', va='center')
                    if i==3:
                        plt.scatter(1-0.9193, 1.0, facecolors='none', edgecolors='green', linewidths=1, s=100, label='SoftWare Rabbit')
                        plt.text(1-0.9193, 1.0, 'S', color='green', fontsize=8, ha='center', va='center')
                        # for idx, (human_id, v) in enumerate(human_results.items()):
                            # plt.scatter(1-v['Rabbit']['Specificity'], v['Rabbit']['Sensitivity'], facecolors='none', edgecolors='green', linewidths=1, s=100,label=None if idx>0 else 'Experts 1~6 Rabbit')
                            # plt.text(1-v['Rabbit']['Specificity'], v['Rabbit']['Sensitivity'], str(idx+1), color='green', fontsize=8, ha='center', va='center')
                    
                    plt.legend(loc="lower right")
                    fig.savefig(f'/data/taoranyi/temp_for_wulianjun/auc_software_{i}.svg')


                
                # for qq in range(6):
                #     fig = plt.figure()
                #     for i,(info,color) in enumerate(zip(infos,colors)):
                #         aa = list(filter(lambda x: x['species'] == info or info in x['slide_name'],inputs_info))
                #         indices = np.array([x['id'] for x in aa],dtype=np.int64)
                #         inp = np.array(inputs_all)[indices].tolist()
                #         outscore = np.array(outputs_scores)[indices].tolist()
                #         fpr1, tpr1, thresholds1 = roc_curve(inp, outscore)
                #         specificitys = []
                #         sensitivitys = []
                #         y_true = np.array(inp)
                #         outscore = np.array(outscore)
                #         weight = sum(y_true)/len(y_true)
                #         thresholds = thresholds1[1:]
                #         num_thresholds = len(thresholds)
                #         specificities = np.zeros(num_thresholds)
                #         sensitivities = np.zeros(num_thresholds)
                #         num_cores = 16  # Number of CPU cores to use
                #         results = Parallel(n_jobs=num_cores)(
                #             delayed(calculate_metrics)(threshold, y_true, outscore, weight)
                #             for threshold in tqdm(thresholds)
                #         )
                #         specificities, sensitivities = zip(*results)
                #         specificitys = np.array(specificities)
                #         sensitivitys = np.array(sensitivities)
                #         roc_auc1 = auc(1-np.array(specificitys), tpr1[1:])
                #         plt.plot(1-np.array(specificitys), tpr1[1:], color=color, label='Weighted ROC curve {} (AUC = {:.2f})'.format(infos[i],roc_auc1))
                #         plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
                #         plt.xlim([xlim_min, xlim_max])
                #         plt.ylim([ylim_min, ylim_max])
                #         plt.xlabel('1-Specificity (%)')
                #         plt.ylabel('Sensitivity (%)')
                #         plt.title(f'Weighted Receiver Operating Characteristic / Expert {qq+1}')
                #         if i==0:
                #             # plt.scatter(1-0.7017, 1.0, facecolors='none', edgecolors='red', linewidths=1, s=100, label='SoftWare Gan')
                #             # plt.text(1-0.7017, 1.0, 'S', color='red', fontsize=8, ha='center', va='center')
                #             for idx, (human_id, v) in enumerate(human_results.items()):
                #                 if idx == qq:
                #                     plt.scatter(1-v['LLC']['Specificity'], v['LLC']['Sensitivity'], facecolors='none', edgecolors='darkorange', linewidths=1, s=100,label= f'Experts {qq+1} LLC')
                #                     plt.text(1-v['LLC']['Specificity'], v['LLC']['Sensitivity'], str(idx+1), color='darkorange', fontsize=8, ha='center', va='center')
                #         if i==1:
                #             # plt.scatter(1-0.7621, 1.0, facecolors='none', edgecolors='blue', linewidths=1, s=100, label='SoftWare Liu')
                #             # plt.text(1-0.7621, 1.0, 'S', color='blue', fontsize=8, ha='center', va='center')
                #             for idx, (human_id, v) in enumerate(human_results.items()):
                #                 if idx == qq:
                #                     plt.scatter(1-v['Liu']['Specificity'], v['Liu']['Sensitivity'], facecolors='none', edgecolors='blue', linewidths=1, s=100,label= f'Experts {qq+1} Liu')
                #                     plt.text(1-v['Liu']['Specificity'], v['Liu']['Sensitivity'], str(idx+1), color='blue', fontsize=8, ha='center', va='center')
                #         if i==2:
                #             # plt.scatter(1-0.7656, 1.0, facecolors='none', edgecolors='darkorange', linewidths=1, s=100, label='SoftWare LLC')
                #             # plt.text(1-0.7656, 1.0, 'S', color='darkorange', fontsize=8, ha='center', va='center')
                #             for idx, (human_id, v) in enumerate(human_results.items()):
                #                 if idx == qq:
                #                     plt.scatter(1-v['Gan']['Specificity'], v['Gan']['Sensitivity'], facecolors='none', edgecolors='red', linewidths=1, s=100,label= f'Experts {qq+1} Gan')
                #                     plt.text(1-v['Gan']['Specificity'], v['Gan']['Sensitivity'], str(idx+1), color='red', fontsize=8, ha='center', va='center')
                #         if i==3:
                #             # plt.scatter(1-0.9193, 1.0, facecolors='none', edgecolors='green', linewidths=1, s=100, label='SoftWare Rabbit')
                #             # plt.text(1-0.9193, 1.0, 'S', color='green', fontsize=8, ha='center', va='center')
                #             for idx, (human_id, v) in enumerate(human_results.items()):
                #                 if idx == qq:
                #                     plt.scatter(1-v['Rabbit']['Specificity'], v['Rabbit']['Sensitivity'], facecolors='none', edgecolors='green', linewidths=1, s=100,label= f'Experts {qq+1} Rabbit')
                #                     plt.text(1-v['Rabbit']['Specificity'], v['Rabbit']['Sensitivity'], str(idx+1), color='green', fontsize=8, ha='center', va='center')
                        
                #         legend = plt.legend(loc="lower right")
                #         for label in legend.get_texts():
                #             label.set_fontsize(6)
                    

                #     fig.savefig(f'/data/taoranyi/temp_for_wulianjun/auc_all_expert_{qq}.svg')
                # plt.legend(loc="lower right")
                # fig.savefig('/data/taoranyi/temp_for_wulianjun/auc_400_split.png')

            # f_txt.close()
            # f_txt2.close()
            get_metric(inputs_all,outputs_scores,inputs_info)
            # show_split_auc(inputs_all,outputs_scores,inputs_info,0,1,0.0,1.05)
            # show_auc(inputs_all,outputs_scores)
        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg
        # cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.OptimizationHook(
                accumulate_grad_steps=cfg.SOLVER.BATCH_SUBDIVISIONS,
                grad_clipper=None,
                mixed_precision=cfg.TRAINER.FP16.ENABLED
            ),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer,
                cfg.SOLVER.CHECKPOINT_PERIOD,
                max_iter=self.max_iter,
                max_epoch=self.max_epoch
            ))

        def test_and_save_results():
            evaluation_type = self.cfg.TEST.get('EVALUATION_TYPE',None)

            if evaluation_type is not None:
                evaluator = EVALUATOR.get(evaluation_type)()
            else:
                evaluator = None
            self._last_eval_results = self.test(self.cfg, self.model,evaluators=evaluator,output_folder=self.cfg.OUTPUT_DIR)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(
                self.build_writers(), period=self.cfg.GLOBAL.LOG_INTERVAL
            ))
            # Put `PeriodicDumpLog` after writers so that can dump all the files,
            # including the files generated by writers

        return ret

    def build_writers(self):
        """
        Build a list of :class:`EventWriter` to be used.
        It now consists of a :class:`CommonMetricPrinter`,
        :class:`TensorboardXWriter` and :class:`JSONWriter`.

        Args:
            output_dir: directory to store JSON metrics and tensorboard events
            max_iter: the total number of iterations

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(
                self.max_iter,
                window_size=self.window_size,
                epoch=self.max_epoch,
            ),
            JSONWriter(
                os.path.join(self.cfg.OUTPUT_DIR, "metrics.json"),
                window_size=self.window_size
            )
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        if self.max_epoch is None:
            logger.info("Starting training from iteration {}".format(self.start_iter))
        else:
            logger.info("Starting training from epoch {}".format(self.start_epoch))

        super().train(self.start_iter, self.start_epoch, self.max_iter)

        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`cvpods.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, **kwargs):
        """
        It now calls :func:`cvpods.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer, **kwargs)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`cvpods.data.build_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`cvpods.data.build_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_test_loader(cfg)


    def run_step(self):
        """
        Implement the standard training logic described above.
        """
      
        assert self.model.training, "[IterRunner] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self.epoch += 1
                
            if hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(self.epoch)
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)

        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        if self.teacher_model is not None:
            loss_dict = self.model(data,self.teacher_model)
        else:
            loss_dict = self.model(data)

        self.inner_iter += 1

        losses = sum([
            metrics_value for metrics_value in loss_dict.values()
            if metrics_value.requires_grad
        ])
        self._detect_anomaly(losses, loss_dict)

        self._write_metrics(loss_dict, data_time)

        self.step_outputs = {
            "loss_for_backward": losses,
        }



# fig = plt.figure()
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_value)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend(loc="lower right")