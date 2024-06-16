#! C:\Users\92429\Anaconda3\python.exe
# -*- encoding: utf-8 -*-

import os
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import dataset, Sampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload
from abc import abstractmethod
# from torch.utils.data.sampler import RandomSampler, Sampler
import json
import numpy as np
from tqdm import tqdm

from datetime import datetime

# 获取当前时间
now = datetime.now()

# 将时间格式化为字符串
time_str = now.strftime("%Y%m%d_%H%M")

import logging
logging.basicConfig(
    filename="./log/esl{}.log".format(time_str),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)s:%(funcName)s] - %(message)s",
)

class Trainer(object):
    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 output_dir:str,
                 training_dataset: Dataset,
                 valid_dataset: Dataset,
                 test_dataset: Dataset,
                 metrics_key: str,
                 epochs: int,
                 batch_size: int,

                 batch_forward_func: Callable[[Tuple[torch.Tensor, ...], 'Trainer'], Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], Union[torch.Tensor, Tuple[torch.Tensor, ...]]]],
                 batch_cal_loss_func: Callable[[Union[torch.Tensor, Tuple[torch.Tensor, ...]], Union[torch.Tensor, Tuple[torch.Tensor, ...]], 'Trainer'], torch.Tensor],
                 batch_metrics_func: Callable[[Union[torch.Tensor, Tuple[torch.Tensor, ...]], Union[torch.Tensor, Tuple[torch.Tensor, ...]], Dict[str, Union[int, torch.Tensor]], 'Trainer'], Tuple[Dict[str, Union[int, torch.Tensor]], Dict[str, Union[int, torch.Tensor]]]],
                 metrics_cal_func: Callable[[Dict[str, Union[int, torch.Tensor]], 'Trainer'], Dict[str, int]],
                 device: torch.device = torch.device("cpu"),
                 resume_path: str = None,
                 start_epoch: int = 0,
                 train_dataset_sampler: Sampler = None,
                 valid_dataset_sampler:Sampler=None,
                 collate_fn=None,
                 valid_step=1,
                 lr_scheduler=None,
                 gradient_accumulate=1,
                 n_split=-1,
                 save_model:bool=False
                 ) -> None:
        """
        trainer.variables用来保存valid,query的中间变量等
        batch_forward_func:输入一个batch的数据,返回labels,preds
        batch_cal_loss_func:输入一个batch的labels,preds,返回loss
        batch_metrics_func:输入一个batch的labels,preds,一个epoch的metrics
        """
        self.variables: Dict[str, Any] = {}

        # dict<epoch,metrics>
        self.epoch_metrics: List[Dict[str, int]] = []
        self.device = device
        self.model = model.to(self.device)
        self.output_dir=output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.optimizer = optimizer
        self.training_dataset = training_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.batch_size = batch_size

        self.collate_fn = collate_fn
        self.batch_forward_func = batch_forward_func
        self.batch_cal_loss_func = batch_cal_loss_func
        self.batch_metrics_func = batch_metrics_func
        self.metrics_cal_func = metrics_cal_func
        self.metrics_key=metrics_key
        self.valid_step=valid_step
        self.lr_scheduler=lr_scheduler
        self.gradient_accumulate=gradient_accumulate
        self.n_split=n_split
        self.save_model=save_model

        if self.lr_scheduler==None:
            self.lr_scheduler=CosineAnnealingLR(self.optimizer,eta_min=1e-7,verbose=True,T_max=40)
        if self.training_dataset != None and len(self.training_dataset) > 0:
            self.training_dataloader = DataLoader(
                self.training_dataset, batch_size=self.batch_size, shuffle= True,\
                      drop_last=False, sampler=train_dataset_sampler,\
                          collate_fn=self.collate_fn)
        if self.valid_dataset != None and len(self.valid_dataset) > 0:
            self.valid_dataloader = DataLoader(
                self.valid_dataset, batch_size=self.batch_size, shuffle=False,\
                    sampler=valid_dataset_sampler,collate_fn=self.collate_fn)
        if self.test_dataset != None and len(self.test_dataset) > 0:
            self.test_dataloader = DataLoader(
                self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn
            )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_epoch = start_epoch
        self.training_stage = None
        # if resume_path != None:
        #     with open(resume_path, "rb") as f:
        #         self.model.load_state_dict(torch.load(f))

    def valid_epoch(self, epoch: int):
        self.training_stage = "valid"
        self.model.eval()
        valid_count = 0
        data_iter = iter(self.valid_dataloader)
        batch_index = 0
        total_loss = 0.0
        metrics = {}
        with tqdm(total=len(self.valid_dataloader),ncols=80) as tqbar:
            with torch.no_grad():
                while True:
                    data = None
                    try:
                        data = next(data_iter)
                    except Exception:
                        break
                    labels, preds = self.batch_forward_func(data, self)
                    metrics, batch_metrics = self.batch_metrics_func(
                        labels, preds, metrics, self)
                    loss, causal_loss = self.batch_cal_loss_func(labels, preds, self)

                    total_loss += loss.item() 
                    batch_index += 1
                    tqbar.update(1)
            self.logger.info("epoch {0} : valid mean loss {1}".format(
                epoch, total_loss/len(self.valid_dataloader)))
            metrics_result = self.metrics_cal_func(metrics, self)
            for k, v in metrics_result.items():
                self.logger.info(
                    "epoch {0} : valid {1}\t{2}".format(epoch, k, v))
        return metrics_result

    def train_epoch(self, epoch: int) -> Dict[str, int]:
        self.training_stage = "train"
        self.model = self.model.to(self.device)
        self.model.train()
        count = 0
        # wsy: 设置进度条宽度为80个字符
        with tqdm(total=len(self.training_dataloader), ncols=80) as tqbar:
            # wsy：转换为可迭代对象
            data_iter = iter(self.training_dataloader)
            batch_index = 0
            metrics = {}
            while True:                   
                data = None
                try:
                    data = next(data_iter)
                    count +=1
                except StopIteration:
                    break
                except Exception as ex:
                    self.logger.warn(ex)
                
                    break

                labels, preds = self.batch_forward_func(data, self)
                loss, causal_loss = self.batch_cal_loss_func(labels, preds, self)
                if count%500 == 0:
                    print(f"训练阶段：n_split:{self.n_split}, epoch:{epoch},all_loss:{loss}, causal_loss:{causal_loss}")
                loss.backward()
                if (batch_index+1)%self.gradient_accumulate==0 or batch_index==len(self.training_dataloader)-1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                metrics, batch_matrics = self.batch_metrics_func(
                    labels, preds, metrics, self)

                # self.logger.info("epoch {0} : batch {1}/{2} mean loss : {3}".format(
                #     epoch, (batch_index+1), len(self.training_dataloader), loss.item()))
                # for k, v in batch_matrics.items():
                #     self.logger.info("epoch {0} : batch {1}/{2} training {3} : {4}".format(
                #         epoch, (batch_index+1), len(self.training_dataloader), k, v))
                batch_index += 1
                tqbar.update(1)
            metrics_result = self.metrics_cal_func(metrics, self)
            if self.lr_scheduler!=None:
                self.lr_scheduler.step()
            for k, v in metrics_result.items():
                self.logger.info("epoch {0} : training {1} : {2}".format(epoch, k, v))
            # torch.cuda.empty_cache()
            return metrics_result

    def test_epoch(self, epoch: int):
        self.model = self.model.to(self.device)
        self.model.eval()
        data_iter = iter(self.test_dataloader)
        batch_index = 0
        test_result = None
        with torch.no_grad():
            data = None
            while True:
                try:
                    data = next(data_iter)
                except StopIteration:
                    break
                labels, preds = self.batch_forward_func(data, self)
                if type(preds) == tuple:
                    if test_result == None:
                        temp = []
                        for i, ele in enumerate(preds):
                            if type(ele) == list:
                                temp.append(ele)
                            elif type(ele) == torch.Tensor:
                                temp.append(ele.cpu())
                            elif type(ele) == np.ndarray:
                                temp.append(ele)
                            else:
                                raise RuntimeError("preds元素类型错误")
                        test_result = tuple(temp)
                    else:
                        temp = []
                        for i, ele in enumerate(preds):
                            if type(ele) == list:
                                temp.append(test_result[i]+ele)
                            elif type(ele) == torch.Tensor:
                                temp.append(
                                    torch.cat([test_result[i], ele.cpu()], dim=0))
                            elif type(ele) == np.ndarray:
                                temp.append(
                                    np.append(test_result[i], ele, axis=0)
                                )

                        test_result = tuple(temp)
                elif type(preds) == torch.Tensor:
                    if test_result == None:
                        test_result = preds.cpu()
                    else:
                        test_result = torch.cat(
                            [test_result, preds.cpu()], dim=0)
                elif type(preds) == np.ndarray:
                    if(test_result) == None:
                        test_result = preds
                    else:
                        test_result: np.ndarray = np.append(
                            test_result, preds, axis=0)
                elif type(preds) == list:
                    if test_result == None:
                        test_result = preds
                    else:
                        test_result = test_result+preds
                else:
                    raise RuntimeError("preds元素类型错误")

        return test_result

    def get_best_epoch(self)->Optional[int]:
        if len(self.epoch_metrics)==0:
            return None
        else:
            res=0
            for i in range(len(self.epoch_metrics)):
                if self.epoch_metrics[i][self.metrics_key] > \
                    self.epoch_metrics[res][self.metrics_key]:
                    res=i
            return res

    def train(self):

        for epoch in range(self.start_epoch, self.epochs):

            self.logger.info("开始训练 epoch : {}".format(epoch))
            self.train_epoch(epoch)
            if (epoch+1)%self.valid_step==0:
                self.epoch_metrics.append(self.valid_epoch(epoch))

        res=self.get_best_epoch()
        print("..............", self.n_split, "..............")
        print("best_epoch:",res,"  precision:", self.epoch_metrics[res]["precision"], "  recall:", self.epoch_metrics[res]["recall"], "  f1:", self.epoch_metrics[res]["f1"])
        with open(os.path.join(self.output_dir, "{0}_metrics_{1}.txt".format("ESL", time_str)), "a") as f:
            f.write("best_epoch:{0}\n".format(res))
            f.write(json.dumps(self.epoch_metrics[res]))
            f.write("\n")