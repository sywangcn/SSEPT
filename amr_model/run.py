import argparse
import json
import pickle
import random
import sys
from typing import Any, Dict
import warnings
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizer
from typing_extensions import Literal

from amr_model.data_process import (baseline_init_tokenizer,baseline_preprocess_data, BaselineCollator)
from amr_model.model.Without_AMRModule import (BaselineFinetuningModel, get_optimizer, batch_forward_func, batch_cal_loss_func,\
                                                     batch_metrics_func, metrics_cal_func)

from trainer import Trainer

warnings.filterwarnings("ignore")

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../data/shortest_path.json")
parser.add_argument("--split_n", type=int, default=5)
parser.add_argument("--main_device", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument("--learning_rate", type=float, default=0.00001)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--mlm_name_or_path", type=str, default="roberta-base")
parser.add_argument("--gradient_accumulate",type=int,default=1)
parser.add_argument("--seed", type=int, default=1111)
parser.add_argument("--mlm_dropout", type=float, default=0.3)
parser.add_argument("--rnn_input", type=int, default=768)
parser.add_argument("--rnn_hidden", type=int, default=384)
parser.add_argument("--rnn_num_layers", type=int, default=1)
parser.add_argument("--mlp_input", type=int, default=300)

# parser.add_argument("--device_ids", type=str, default="0,1")
if __name__ == '__main__':
    args=parser.parse_args()
    """一些常规的设置"""
    set_random_seed(args.seed)
    dev = torch.device(args.main_device)
    # device_ids=list(map(lambda x:int(x),args.device_ids.split(",")))
    batch_size=args.batch_size
    # num_workers=args.num_workers
    learning_rate=args.learning_rate
    epochs=args.epochs
    mlm_type=args.mlm_name_or_path
    data_path=args.data_path
    split_n=args.split_n
    gradient_accumulate=args.gradient_accumulate

    for arg in args._get_kwargs():
        print(arg)

    metrics = []
    raw_data: Dict[str, Any] = {}
    with open(data_path, "r") as f:
        raw_data = json.load(f)
    kfold = KFold(n_splits=split_n, shuffle=False)

    tokenizer = baseline_init_tokenizer(mlm_type, "prompt_tuning/mlm")
    j=0
    for train_indexs, valid_indexs in kfold.split(raw_data):
        """Timebank十折交叉验证  Eventstory五折交叉验证"""
        j=j+1
        train_raw_dataset = [raw_data[i] for i in train_indexs]
        valid_raw_dataset = [raw_data[i] for i in valid_indexs]

        # tokenizer = RobertaTokenizer.from_pretrained("prompt_tuning/mlm")
        mlm = RobertaForMaskedLM.from_pretrained(mlm_type)
        train_dataset = []
        valid_dataset = []
        train_count = 0
        valid_count = 0
        for data in train_raw_dataset:
            train_dataset.extend(baseline_preprocess_data(data, tokenizer, "train", train_count))

        for data in valid_raw_dataset:
            # data = valid_data_preprocess(data)
            valid_count+=1
            valid_dataset.extend(baseline_preprocess_data(data, tokenizer, "train", valid_count))

        model = BaselineFinetuningModel(mlm_type, args)
        optimizer = get_optimizer(model, learning_rate)

        collator = BaselineCollator(tokenizer)

        train_dataset_sampler = SequentialSampler(train_dataset)
        valid_dataset_sampler = SequentialSampler(valid_dataset)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            output_dir="C:\wsy\PycharmProjects\PCLearning\prompt_tuning\saved",
            training_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=None,
            metrics_key="f1",
            epochs=epochs,
            batch_size=batch_size,
            batch_forward_func=batch_forward_func,
            batch_cal_loss_func=batch_cal_loss_func,
            batch_metrics_func=batch_metrics_func,
            metrics_cal_func=metrics_cal_func,
            collate_fn=collator,
            device=dev,
            train_dataset_sampler=None,
            valid_dataset_sampler=None,
            valid_step=1,
            start_epoch=0,
            gradient_accumulate=gradient_accumulate,
            n_split=j
        )

        trainer.train()
        metrics.append(trainer.epoch_metrics[trainer.get_best_epoch()])

        break


    print(metrics)
    print()
    print(sum(list(map(lambda x: x["f1"], metrics))) / len(metrics))
