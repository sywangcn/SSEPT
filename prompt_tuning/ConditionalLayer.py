#!/home/zhouheng/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@文件    :baseline_finetuning_model.py
@说明    :
@时间    :2021/11/27 16:02:31
@作者    :周恒
@版本    :1.0
Temporal_lm_head_output = self.Causal_lm_head(sequence_output)
'''

from typing import Callable, Dict, Optional, Tuple, Union
import torch
from torch.functional import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.sparse import Embedding
from torch.types import Number
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta import RobertaForMaskedLM, RobertaModel, RobertaConfig
from transformers.models.bert import BertForMaskedLM, BertModel, BertConfig
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.metrics import confusion_matrix

# from data_process import T1,T2,T3,T4,T5,T6,CAUSEOF,NOTCAUSEOF


# 条件层归一化
class ConditionalLayer1(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(ConditionalLayer1, self).__init__()
        self.eps = eps
        self.gamma_dense = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_dense = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gamma = torch.nn.Parameter(torch.ones(hidden_size))
        self.beta = torch.nn.Parameter(torch.zeros(hidden_size))

        torch.nn.init.zeros_(self.gamma_dense.weight)
        torch.nn.init.zeros_(self.beta_dense.weight)

    def forward(self, x, condition):
        '''

        :param x: [b, t, e]
        :param condition: [b, e]
        :return:
        '''
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        gamma = self.gamma_dense(condition) + self.gamma
        beta = self.beta_dense(condition) + self.beta
        x = gamma * (x - mean) / (std + self.eps) + beta
        return x


class BaselineFinetuningModel(torch.nn.Module):
    def __init__(self, mlm_name_or_path: str):
        super().__init__()
        ## Union增加代码的灵活性，变量类型是其中的任一个；  加载适用于掩码语言建模的模型
        mlm_for_maskedlm: Union[BertForMaskedLM, RobertaForMaskedLM] = AutoModelForMaskedLM.from_pretrained(
            mlm_name_or_path)
        self.mlm_config: Union[BertConfig, RobertaConfig] = AutoConfig.from_pretrained(mlm_name_or_path)
        self.hidden_dim = self.mlm_config.hidden_size
        if hasattr(mlm_for_maskedlm, "bert"):
            assert type(mlm_for_maskedlm) == BertForMaskedLM
            self.mlm_type = "bert"
            self.mlm: BertModel = mlm_for_maskedlm.bert
            self.Causal_lm_head = mlm_for_maskedlm.cls.predictions.transform
            self.Temporal_lm_head = mlm_for_maskedlm.cls.predictions.transform
            # self.lm_decoder=mlm_for_maskedlm.cls.predictions.decoder

        elif hasattr(mlm_for_maskedlm, "roberta"):
            assert type(mlm_for_maskedlm) == RobertaForMaskedLM
            self.mlm_type = "roberta"
            self.mlm: RobertaModel = mlm_for_maskedlm.roberta

            # dense1=mlm_for_maskedlm.lm_head.dense
            # gelu1=torch.nn.GELU()
            # layer_norm1= mlm_for_maskedlm.lm_head.layer_norm


            # self.Causal_lm_head = torch.nn.Sequential(
            #     dense1,
            #     gelu1,
            #     layer_norm1
            # )
            """
                using two prompts
            """
            # dense2 = mlm_for_maskedlm.lm_head.dense
            # gelu2 = torch.nn.GELU()
            # layer_norm2= mlm_for_maskedlm.lm_head.layer_norm
            # self.Temporal_lm_head = torch.nn.Sequential(
            #     dense2,
            #     gelu2,
            #     layer_norm2
            # )
            self.lm_decoder=mlm_for_maskedlm.lm_head.decoder
        else:
            raise NotImplemented("目前仅支持bert,roberta")

        self.dropout = torch.nn.Dropout()

        """1+6,第一个为占位 -> 1+8"""
        self.new_embedding = torch.nn.Embedding(9, self.hidden_dim)

        self.fusion1_fc=torch.nn.Linear(self.hidden_dim, 256)
        self.Causal_classification = torch.nn.Linear(256, 3)
        """"
            using two prompts
        """
        self.fusion2_fc = torch.nn.Linear(self.hidden_dim, 256)
        self.Temporal_classification = torch.nn.Linear(256, 3)


        self.ConditionIntegrator1 = ConditionalLayer1(self.hidden_dim)
        self.ConditionIntegrator2 = ConditionalLayer1(self.hidden_dim)

    def forward_one_sentence(
            self,
            input_ids: torch.Tensor,  # batch_size,sequence_length
            masks: torch.Tensor,  # batch_size,sequence_length
            input_ids_for_new: torch.Tensor,  # 用于指示输入序列中属于6个新embedding的部分 值为[0,6]
            mask_positions: torch.Tensor
    ):
        batch_size, sequence_length = input_ids.shape

        """扩展input_ids_for_new,使之为batch_size*sequence*768"""

        input_ids_for_new_repeated = input_ids_for_new.reshape([batch_size, sequence_length, 1]).repeat(1, 1,
                                                                                                        self.hidden_dim)

        # batch_size*sequence*768

        raw_embeddings: torch.Tensor = self.mlm.embeddings.word_embeddings(input_ids)
        """把raw_embeddings中属于新加的8个token的部分置零"""
        zeros = torch.zeros(raw_embeddings.shape, dtype=raw_embeddings.dtype, device=raw_embeddings.device)
        raw_embeddings = torch.where(~input_ids_for_new_repeated.bool(), raw_embeddings, zeros)

        # batch_size*sequence*768
        new_embeddings: torch.Tensor = self.new_embedding(input_ids_for_new)
        """把new_embeddings中属于新加的8个token以外的部分置零"""
        new_embeddings = torch.where(input_ids_for_new_repeated.bool(), new_embeddings, zeros)

        input_embedding = new_embeddings + raw_embeddings

        mlm_output = self.mlm(inputs_embeds=input_embedding, attention_mask=masks)

        sequence_output = mlm_output[0]  # batch_size*sequence_length*768

        # Causal_lm_head_output = self.Causal_lm_head(sequence_output)  # batch_size*sequence_length*768
        # Temporal_lm_head_output = self.Causal_lm_head(sequence_output)

        Causal_masked_positions = torch.zeros((batch_size,1)).to(torch.device(0)).long()
        Temporal_masked_positions = torch.zeros((batch_size, 1)).to(torch.device(0)).long()

        for i in range(0, batch_size, 2):
            Causal_masked_positions[i,:]=mask_positions[i,1]
            Temporal_masked_positions[i,:]=mask_positions[i+1,1]

        Causal_masked_features = sequence_output[torch.arange(batch_size), Causal_masked_positions.reshape([-1]), :]  # batch_size*768

        Temporal_masked_features = sequence_output[torch.arange(batch_size), Temporal_masked_positions.reshape([-1]), :]  # batch_size*768

        if self.training:
            Causal_masked_features = self.dropout(Causal_masked_features)
            Temporal_masked_features = self.dropout(Temporal_masked_features)


        """
            information fusion
        """
        dimi=Causal_masked_features.size(-1)

        fusion_value1=self.ConditionIntegrator1(Causal_masked_features, Temporal_masked_features)
        fusion_value2 = self.ConditionIntegrator2(Temporal_masked_features, Causal_masked_features)

        fusion1_value=self.fusion1_fc(fusion_value1)
        fusion2_value = self.fusion2_fc(fusion_value2)

        Causal_prediction = self.Causal_classification(fusion1_value)  # batch_size*2
        Temporal_prediction = self.Temporal_classification(fusion2_value)


        return Causal_prediction, Temporal_prediction

    def forward(
            self,
            input_ids1: torch.Tensor,  # batch_size,sequence_length
            masks1: torch.Tensor,  # batch_size,sequence_length
            input_ids_for_new1: torch.Tensor,  # 用于指示输入序列中属于6个新embedding的部分 值为[0,6]
            mask_positions1: torch.Tensor,
            input_ids2: torch.Tensor,  # batch_size,sequence_length
            masks2: torch.Tensor,  # batch_size,sequence_length
            input_ids_for_new2: torch.Tensor,  # 用于指示输入序列中属于6个新embedding的部分 值为[0,6]
            mask_positions2: torch.Tensor
    ):
        pred1, pred3 = self.forward_one_sentence(input_ids1, masks1, input_ids_for_new1, mask_positions1)
        pred2, pred4 = self.forward_one_sentence(input_ids2, masks2, input_ids_for_new2, mask_positions2)
        return pred1, pred3, pred2, pred4


def get_optimizer(model: BaselineFinetuningModel, lr: float):
    # optimizer=torch.optim.AdamW([
    # {"params":model.new_embedding.parameters()},
    # # {"params":model.fc.parameters()}
    # {"params":model.classification.parameters()}
    # ],lr=lr)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.mlm.parameters(), "lr": lr / 10},
            # {"params": model.Causal_lm_head.parameters(), "lr": lr / 10},
            # {"params": model.Temporal_lm_head.parameters(), "lr": lr / 10},
            {"params": model.new_embedding.parameters()},
            {"params": model.ConditionIntegrator1.parameters()},
            {"params": model.ConditionIntegrator2.parameters()},
            {"params": model.fusion1_fc.parameters()},
            {"params": model.fusion2_fc.parameters()},
            {"params": model.Causal_classification.parameters()},
            {"params": model.Temporal_classification.parameters()}
        ],
        lr=lr
    )
    return optimizer


def batch_forward_func(batch_data: Tuple[torch.Tensor, ...], trainer):
    # input_ids1, masks1, input_ids_for_new1, mask_pos1, labels1, \
    #     input_ids2, masks2, input_ids_for_new2, mask_pos2, labels2, batch_signals = batch_data
    input_ids1, masks1, input_ids_for_new1, mask_pos1, labels1, labels3, \
        input_ids2, masks2, input_ids_for_new2, mask_pos2, labels2, labels4=batch_data

    input_ids1, \
        masks1, \
        input_ids_for_new1, \
        mask_pos1, \
        labels1, \
        labels3, \
        input_ids2, \
        masks2, \
        input_ids_for_new2, \
        mask_pos2, \
        labels2, \
        labels4, \
        = \
        input_ids1.to(trainer.device), \
            masks1.to(trainer.device), \
            input_ids_for_new1.to(trainer.device), \
            mask_pos1.to(trainer.device), \
            labels1.to(trainer.device), \
            labels3.to(trainer.device), \
            input_ids2.to(trainer.device), \
            masks2.to(trainer.device), \
            input_ids_for_new2.to(trainer.device), \
            mask_pos2.to(trainer.device), \
            labels2.to(trainer.device), \
            labels4.to(trainer.device), \

    prediction = trainer.model(
        input_ids1,
        masks1,
        input_ids_for_new1,
        mask_pos1,
        input_ids2,
        masks2,
        input_ids_for_new2,
        mask_pos2,
    )

    return (labels1, labels3, labels2, labels4), prediction


def batch_cal_loss_func(labels: Tuple[torch.Tensor, ...], preds: Tuple[torch.Tensor, ...], trainer):
    labels1, labels3, labels2, labels4 = labels
    pred1, pred3, pred2, pred4 = preds

    return F.cross_entropy(pred1, labels1, reduction="mean") + F.cross_entropy(pred2, labels2, reduction="mean") \
        + F.cross_entropy(pred3, labels3, reduction="mean")+ F.cross_entropy(pred4, labels4, reduction="mean")




def batch_metrics_func(labels: Tuple[torch.Tensor, ...], preds: Tuple[torch.Tensor, ...], metrics: Dict[str, Number],
                       trainer):
    labels1, labels3, labels2, labels4 = labels
    pred1, pred3, pred2, pred4 = preds
    # batch_signals = batch_signals.cpu()

    cause_preds1 = torch.argmax(pred1, dim=1).reshape([-1]).cpu()
    cause_preds2 = torch.argmax(pred2, dim=1).reshape([-1]).cpu()
    # cause_preds = torch.logical_or(cause_preds1, cause_preds2).long()
    causes1 = labels1.cpu()
    causes2 = labels2.cpu()
    """ 
        if the predict has one of the two is correct, the result is correct
    """
    # cause_preds=torch.where(cause_preds1!=causes1, cause_preds2, cause_preds1)
    # causes=torch.where(cause_preds1!=causes1, causes2, causes1)

    # trainer.logger.info("labels: \n{}".format(causes))
    # trainer.logger.info("preds: \n{}".format(cause_preds))

    causes=causes1
    cause_preds=cause_preds1

    precision = precision_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")
    recall = recall_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")
    f1 = f1_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")

    batch_metrics = {"precision": precision, "recall": recall, "f1": f1}
    if "labels" in metrics:
        metrics["labels"] = torch.cat([metrics["labels"], causes], dim=0)
    else:
        metrics["labels"] = causes
    if "preds" in metrics:
        metrics["preds"] = torch.cat([metrics["preds"], cause_preds], dim=0)
    else:
        metrics["preds"] = cause_preds

    return metrics, batch_metrics


def metrics_cal_func(metrics: Dict[str, torch.Tensor]):
    causes = metrics["labels"]
    cause_preds = metrics["preds"]


    precision = precision_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")
    recall = recall_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")
    f1 = f1_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")


    cm = confusion_matrix(causes.numpy(), cause_preds.numpy())
    print(cm)

    res = {"precision": precision, "recall": recall, "f1": f1}
    return res
