#!/home/zhouheng/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@文件    :baseline_finetuning_model.py
@说明    :
@时间    :2024/1/13 21:10
@作者    :王松阳
@版本    :1.0
@修改信息   ：删除因果相关性的两个向量定义；删除优化器的定义；删除AMR序列的向量的融合；删除因果相关性的判断损失
'''

from typing import Callable, Dict, Optional, Tuple, Union
import torch
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax
from torch import nn
from torch.functional import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.sparse import Embedding
from torch.types import Number
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoConfig

from transformers.models.roberta import RobertaForMaskedLM, RobertaModel, RobertaConfig
from transformers.models.bert import BertForMaskedLM, BertModel, BertConfig
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# from data_process import T1,T2,T3,T4,T5,T6,CAUSEOF,NOTCAUSEOF


# 位置编码
class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.tensor(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)



class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, lengths):
        # 对输入序列进行打包
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # 初始化隐状态和细胞状态
        h0 = torch.zeros(self.num_layers * (2 if self.lstm.bidirectional else 1), x.batch_sizes[0],
                         self.hidden_size).to(x.data.device)
        c0 = torch.zeros(self.num_layers * (2 if self.lstm.bidirectional else 1), x.batch_sizes[0],
                         self.hidden_size).to(x.data.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 将打包的序列转回原来的形式
        out, _ = pad_packed_sequence(out, batch_first=True)

        # 获取每个序列中最后一个有效元素的输出
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(1).to("cuda:0")
        last_output = out.gather(1, idx).squeeze(1)

        return last_output


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
    def __init__(self, mlm_name_or_path: str, args):
        super().__init__()
        self.args = args
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

            dense1 = mlm_for_maskedlm.lm_head.dense
            gelu1 = torch.nn.GELU()
            layer_norm1 = mlm_for_maskedlm.lm_head.layer_norm

            self.Causal_lm_head = torch.nn.Sequential(
                dense1,
                gelu1,
                layer_norm1
            )
            """ using two prompts"""
            self.lm_decoder = mlm_for_maskedlm.lm_head.decoder
        else:
            raise NotImplemented("目前仅支持bert,roberta")

        self.dropout = torch.nn.Dropout(p=self.args.mlm_dropout)
        """1+6,第一个为占位 -> 1+8"""
        self.new_embedding = torch.nn.Embedding(9, self.hidden_dim)

        # 定义RNN
        self.rnn = RNNModel(input_size=self.args.rnn_input, hidden_size=self.args.rnn_hidden,
                            num_layers=self.args.rnn_num_layers, bidirectional=True)
        self.m = nn.GELU()
        # 定义2个条件归一化
        self.ConditionIntegrator1 = ConditionalLayer1(self.hidden_dim)
        self.ConditionIntegrator2 = ConditionalLayer1(self.hidden_dim)
        # 定义分类器
        fusion1_fc = torch.nn.Linear(self.hidden_dim, self.args.mlp_input)
        Causal_classification = torch.nn.Linear(self.args.mlp_input, 3)
        fusion2_fc = torch.nn.Linear(self.hidden_dim, self.args.mlp_input)
        Temporal_classification = torch.nn.Linear(self.args.mlp_input, 3)
        self.Causal_MLP = torch.nn.Sequential(
            fusion1_fc,
            Causal_classification
        )
        self.Temporal_MLP = torch.nn.Sequential(
            fusion2_fc,
            Temporal_classification
        )

    def forward_one_sentence(
            self,
            input_ids: torch.Tensor,  # batch_size,sequence_length
            masks: torch.Tensor,  # batch_size,sequence_length
            input_ids_for_new: torch.Tensor,  # 用于指示输入序列中属于6个新embedding的部分 值为[0,6]
            mask_positions: torch.Tensor,
            tokenize_ins_index: torch.Tensor
    ):
        # 三维列表所有元素值都加1，因为tokenize会在句首添加<s>，句尾添加</s>
        tokenize_ins_index = [[[j + 1 for j in k] for k in i] for i in tokenize_ins_index]

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

        # 获取backbone模型输出
        mlm_output = self.mlm(inputs_embeds=input_embedding, attention_mask=masks)
        sequence_output = mlm_output[0]  # batch_size*sequence_length*768
        Causal_lm_head_output = self.Causal_lm_head(sequence_output)  # batch_size*sequence_length*768
        # Temporal_lm_head_output = self.Causal_lm_head(sequence_output)

        # 获取因果和时序mask位置向量,作为attention机制的q
        Causal_masked_positions = torch.zeros((batch_size, 1)).to(torch.device(0)).long()
        Temporal_masked_positions = torch.zeros((batch_size, 1)).to(torch.device(0)).long()
        for i in range(0, batch_size, 1):
            Causal_masked_positions[i, :] = mask_positions[2 * i, 1]
            Temporal_masked_positions[i, :] = mask_positions[2 * i + 1, 1]
        Causal_masked_features = Causal_lm_head_output[torch.arange(batch_size), Causal_masked_positions.reshape([-1]),
                                 :]  # batch_size*768
        Temporal_masked_features = Causal_lm_head_output[torch.arange(batch_size),
                                   Temporal_masked_positions.reshape([-1]), :]  # batch_size*768

        if self.training:
            Causal_masked_features = self.dropout(Causal_masked_features)
            Temporal_masked_features = self.dropout(Temporal_masked_features)

        # 获取新序列向量
        new_sequences = []
        sequence_lengths = []

        for batch_idx, word_indices in enumerate(tokenize_ins_index):
            new_sequence = []  # 用于保存当前样本的新序列

            # word_index是一个单词，word_indices是一个路径
            for i, word_index in enumerate(word_indices):
                # 从 Causal_lm_head_output 中获取当前单词的向量
                word_vector = Causal_lm_head_output[batch_idx, word_index, :]
                # 如果一个单词被tokenize为多个部分，我们将这些部分的向量相加并求平均
                word_vector = word_vector.mean(dim=0)
                new_sequence.append(word_vector)
            embeddings = torch.stack(new_sequence)
            new_sequences.append(embeddings)
            sequence_lengths.append(len(new_sequence))

        padded_sequences = pad_sequence(new_sequences, batch_first=True)
        sequence_lengths = torch.LongTensor(sequence_lengths)

        # 使用rnn
        output_rnn = self.rnn(padded_sequences, sequence_lengths)
        Causal_masked_features = Causal_masked_features + output_rnn
        Temporal_masked_features = Temporal_masked_features + output_rnn

        # 条件归一化
        Causal_masked_features = self.m(Causal_masked_features)
        Temporal_masked_features = self.m(Temporal_masked_features)

        fusion_value1 = self.ConditionIntegrator1(Causal_masked_features, Temporal_masked_features)
        fusion_value2 = self.ConditionIntegrator2(Temporal_masked_features, Causal_masked_features)
        # 分类器
        Causal_prediction = self.Causal_MLP(fusion_value1)
        Temporal_prediction = self.Temporal_MLP(fusion_value2)
        Iscausality_label = []
        return Causal_prediction, Temporal_prediction, Iscausality_label

    def forward(
            self,
            input_ids1: torch.Tensor,  # batch_size,sequence_length
            masks1: torch.Tensor,  # batch_size,sequence_length
            input_ids_for_new1: torch.Tensor,  # 用于指示输入序列中属于6个新embedding的部分 值为[0,6]
            mask_positions1: torch.Tensor,
            tokenize_1t02_ins_index: torch.Tensor,
            input_ids2: torch.Tensor,  # batch_size,sequence_length
            masks2: torch.Tensor,  # batch_size,sequence_length
            input_ids_for_new2: torch.Tensor,  # 用于指示输入序列中属于6个新embedding的部分 值为[0,6]
            mask_positions2: torch.Tensor,
            tokenize_2t01_ins_index: torch.Tensor

    ):
        pred1, pred3, pred5 = self.forward_one_sentence(input_ids1, masks1, input_ids_for_new1, mask_positions1, \
                                                        tokenize_1t02_ins_index)
        pred2, pred4, pred6 = self.forward_one_sentence(input_ids2, masks2, input_ids_for_new2, mask_positions2, \
                                                        tokenize_2t01_ins_index)
        return pred1, pred3, pred2, pred4, pred5, pred6


def get_optimizer(model: BaselineFinetuningModel, lr: float):
    optimizer = torch.optim.Adam(
        [
            {"params": model.mlm.parameters(), "lr": lr / 10},
            {"params": model.Causal_lm_head.parameters()},
            {"params": model.new_embedding.parameters()},
            {"params": model.dropout.parameters()},

            {"params": model.rnn.parameters()},

            {"params": model.ConditionIntegrator1.parameters()},
            {"params": model.ConditionIntegrator2.parameters()},
            {"params": model.Causal_MLP.parameters()},
            {"params": model.Temporal_MLP.parameters()}
        ],
        lr=lr
    )
    return optimizer


def batch_forward_func(batch_data, trainer):
    # input_ids1, masks1, input_ids_for_new1, mask_pos1, labels1, \
    #     input_ids2, masks2, input_ids_for_new2, mask_pos2, labels2, batch_signals = batch_data
    input_ids1, masks1, input_ids_for_new1, mask_pos1, labels1, labels3, tokenize_1t02_ins_index, \
        input_ids2, masks2, input_ids_for_new2, mask_pos2, labels2, labels4, tokenize_2t01_ins_index, labels5 = batch_data

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
        labels5 \
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
            labels5.to(trainer.device)
    prediction = trainer.model(
                input_ids1,
                masks1,
                input_ids_for_new1,
                mask_pos1,
                tokenize_1t02_ins_index,
                input_ids2,
                masks2,
                input_ids_for_new2,
                mask_pos2,
                tokenize_2t01_ins_index
            )

    return (labels1, labels3, labels2, labels4, labels5), prediction


def batch_cal_loss_func(labels: Tuple[torch.Tensor, ...], preds: Tuple[torch.Tensor, ...], trainer):
    labels1, labels3, labels2, labels4, labels5 = labels
    pred1, pred3, pred2, pred4, pred5, pred6 = preds
    weights = torch.tensor([0.14, 1.15, 1.0], device="cuda:0")

    Total_Loss = F.cross_entropy(pred1, labels1, reduction="mean", weight=weights) + F.cross_entropy(pred2, labels2,
                                                                                                     reduction="mean",
                                                                                                     weight=weights) \
                 + F.cross_entropy(pred3, labels3, reduction="mean", weight=weights) + F.cross_entropy(pred4, labels4,
                                                                                                       reduction="mean",
                                                                                                       weight=weights)

    return Total_Loss, F.cross_entropy(pred1, labels1, reduction="mean", weight=weights)


def batch_metrics_func(labels: Tuple[torch.Tensor, ...], preds: Tuple[torch.Tensor, ...], metrics: Dict[str, Number],
                       trainer):
    labels1, labels3, labels2, labels4, labels5 = labels
    pred1, pred3, pred2, pred4, pred5, pred6 = preds
    # batch_signals = batch_signals.cpu()

    cause_preds1 = torch.argmax(pred1, dim=1).reshape([-1]).bool().long().cpu()
    cause_preds2 = torch.argmax(pred2, dim=1).reshape([-1]).bool().long().cpu()
    # cause_preds = torch.logical_or(cause_preds1, cause_preds2).long()
    causes1 = labels1.reshape([-1]).bool().long().cpu()
    causes2 = labels2.cpu()
    """ 
        if the predict has one of the two is correct, the result is correct
    """
    # cause_preds=torch.where(cause_preds1!=causes1, cause_preds2, cause_preds1)
    # causes=torch.where(cause_preds1!=causes1, causes2, causes1)

    # trainer.logger.info("labels: \n{}".format(causes))
    # trainer.logger.info("preds: \n{}".format(cause_preds))

    causes = causes1
    cause_preds = cause_preds1

    precision = precision_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")
    recall = recall_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")
    f1 = f1_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")

    if trainer.training_stage == "valid":
        trainer.logger.info("labels: {}".format(causes))
        trainer.logger.info("preds: {}".format(cause_preds))

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


def metrics_cal_func(metrics: Dict[str, torch.Tensor], trainer):
    causes = metrics["labels"]
    cause_preds = metrics["preds"]

    precision = precision_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")
    recall = recall_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")
    f1 = f1_score(causes.numpy(), cause_preds.numpy(), zero_division=0, average="macro")

    cm = confusion_matrix(causes.numpy(), cause_preds.numpy())
    print(cm)
    trainer.logger.info("混淆矩阵: {}".format(cm))
    res = {"precision": precision, "recall": recall, "f1": f1}
    return res
