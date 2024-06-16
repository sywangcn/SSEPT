from typing import Any, Dict, Iterator, List, Optional, Sequence, Sized, Tuple, Union
from dataclasses import asdict, dataclass
from numpy.core.fromnumeric import shape
import torch
from torch.utils.data.sampler import WeightedRandomSampler,Sampler
from transformers import AutoModel,AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
import numpy as np
"""6个指示符"""

"""</t1></t2>指示source event 的位置"""
T1="</t1>"
T2="</t2>"

"""</t3></t4>指示target event 的位置"""
T3="</t3>"
T4="</t4>"

"""</t5></t6>指示Causal Mask的位置"""
T5="</t5>"
T6="</t6>"

"""</t7></t8>指示Temporal Mask的位置"""
T7="</t7>"
T8="</t8>"

"""bert input sequence max length """
MAX_LENTH = 512


def baseline_init_tokenizer(name_or_path: str, save_dir: str) -> Tuple[
    Union[BertModel, RobertaModel], Union[BertTokenizer, RobertaTokenizer]]:
    """初始化分词器,加入8个特殊字符"""

    # mlm:AutoModel=AutoModel.from_pretrained(name_or_path)
    mlm_tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    """分词器增加特殊字符"""
    special_tokens_dict = {"additional_special_tokens": [T1, T2, T3, T4, T5, T6, T7, T8]}
    mlm_tokenizer.add_special_tokens(special_tokens_dict)

    # """预训练模型扩充token embedding,新增加的token embedding是随机初始化的"""
    # mlm.resize_token_embeddings(len(mlm_tokenizer))

    mlm_tokenizer.save_pretrained(save_dir)

    return mlm_tokenizer


@dataclass
class BaselineInputfeature:
    cause1:int
    cause2:int
    prompted_sentence1:str
    prompted_sentence2:str
    tokenize_1t02_ins_index:list
    tokenize_2t01_ins_index:list



def make_prompted(
        tokens: List[str],
        source_start_index: int,
        source_end_index: int,
        target_start_index: int,
        target_end_index: int,
        mask_token: str,
        substr_token_start_index: int,
        substr_token_end_index: int,
) -> Optional[str]:
    """调用前确保prompt后完整的句子分词后长度不会超过512
    """
    new_tokens = tokens.copy()

    # wsy:完整s The event </t1>event1</t2> has the </t5> [Mask] </t6> of the event </t3> event2 </t4>.
    prompted_tokens = \
        new_tokens[substr_token_start_index:substr_token_end_index + 1] + \
        new_tokens[source_start_index:source_end_index + 1] + \
        ["is", T5, mask_token, T6, "influenced by"] + \
        new_tokens[target_start_index:target_end_index + 1] + ["."]

    return " ".join(prompted_tokens)

def baseline_preprocess_data(data: Dict[str, Any], tokenizer: Union[BertTokenizer, RobertaTokenizer]) -> List[
    BaselineInputfeature]:
    """
    :param data:
    :param tokenizer:
    :return:
    tokenize_ins_index=[ [0,1,2],[7,8],...] 保存final_path中节点单词的向量索引
    """
    res = []
    temporality=0
    tok2ori_map = []
    cause = data["relation"]
    event1_start_index = data["event1_start"]
    event1_end_index = data["event1_end"]
    event2_start_index = data["event2_start"]
    event2_end_index = data["event2_end"]
    sentence = data["sentence"]

    tokens = sentence.split()
    # if "Then" and "came" in tokens:
    #     print(tokens)
    final_path_1t02 = data["final_path_1t02"]
    final_path_2t01 = data["final_path_2t01"]
    alignments_keys = data["alignments_keys"]
    instances_index = data["instances_index"]

    tokenize_1t02_ins_index = []
    final_path_1t02_index = []  # 保存instance 在文本中的索引

    tokenize_2t01_ins_index = []
    final_path_2t01_index = []  # 保存instance 在文本中的索引

    t_lists = []
    for ori_i, w in enumerate(tokens):
        if ori_i != 0:
            w = "<s>"+" " + w
            t_lists = tokenizer.tokenize(w)
            for _ in t_lists[1:]:
                tok2ori_map.append(ori_i)
        else:
            for _ in tokenizer.tokenize(w):
                tok2ori_map.append(ori_i)
    # print(tok2ori_map)

    # 1to2
    if len(final_path_1t02) != 0:
        for node in final_path_1t02:
            if node in alignments_keys:
                idx = alignments_keys.index(node)
                final_path_1t02_index.append(instances_index[idx])
            else:
                print("alignments_keys中没有变量{}".format(node))

        # 根据final_path_index中的值，统计每个instance所对应的向量
        for index in final_path_1t02_index:
            index_list = [i for i, x in enumerate(tok2ori_map) if x == index]

            if len(index_list)!=0:
                tokenize_1t02_ins_index.append(index_list)
            else:
                tokenize_1t02_ins_index = []
                for index in (event1_start_index, event2_start_index):
                    index_list = [i for i, x in enumerate(tok2ori_map) if x == index]
                    tokenize_1t02_ins_index.append(index_list)
        # print(tokenize_ins_index)

    else:
        # 如果没有最短路径，那么就直接以2个触发词组成最短路径
        for index in (event1_start_index, event2_start_index):
            index_list = [i for i, x in enumerate(tok2ori_map) if x == index]
            tokenize_1t02_ins_index.append(index_list)

    # 2t01
    if len(final_path_2t01) != 0:
        for node in final_path_2t01:
            if node in alignments_keys:
                idx = alignments_keys.index(node)
                final_path_2t01_index.append(instances_index[idx])
            else:
                print("alignments_keys中没有变量{}".format(node))

        # 根据final_path_index中的值，统计每个instance所对应的向量   len(index_list)==0是什么情况呢？？？
        for index in final_path_2t01_index:
            index_list = [i for i, x in enumerate(tok2ori_map) if x == index]

            if len(index_list)!=0:
                tokenize_2t01_ins_index.append(index_list)
            else:
                tokenize_2t01_ins_index = []
                for index in (event2_start_index, event1_start_index):
                    index_list = [i for i, x in enumerate(tok2ori_map) if x == index]
                    tokenize_2t01_ins_index.append(index_list)
        # print(tokenize_ins_index)

    else:
        # 如果没有最短路径，那么就直接以2个触发词组成最短路径
        for index in (event2_start_index, event1_start_index):
            index_list = [i for i, x in enumerate(tok2ori_map) if x == index]
            tokenize_2t01_ins_index.append(index_list)


    substr_token_start_index = 0
    substr_token_end_index = len(tokens)-1

    prompt1 = make_prompted(
        tokens=tokens,
        source_start_index=event1_start_index,
        source_end_index=event1_end_index,
        target_start_index=event2_start_index,
        target_end_index=event2_end_index,
        mask_token=tokenizer.mask_token,
        substr_token_start_index=substr_token_start_index,
        substr_token_end_index=substr_token_end_index
    )

    # wsy:翻转event1和event2  还需要添加e2至e1的最短路径
    prompt2 = make_prompted(
        tokens=tokens,
        source_start_index=event2_start_index,
        source_end_index=event2_end_index,
        target_start_index=event1_start_index,
        target_end_index=event1_end_index,
        mask_token=tokenizer.mask_token,
        substr_token_start_index=substr_token_start_index,
        substr_token_end_index=substr_token_end_index
    )

    res.append(BaselineInputfeature(cause, -cause, prompt1, prompt2, tokenize_1t02_ins_index, tokenize_2t01_ins_index))
    return res


class BaselineCollator:
    def __init__(self, tokenizer: Union[BertTokenizer, RobertaTokenizer]) -> None:
        self.tokenizer = tokenizer
        self.raw_vacob_size = self.tokenizer.vocab_size

    def __call__(self, data: List[BaselineInputfeature]) -> Tuple[torch.Tensor, ...]:
        batch_size = len(data)
        text1, text2 = [], []
        batch_labels_1, batch_labels_2 = [], [] # 保存因果标签
        batch_labels_5 = []
        tokenize_1t02_ins_index = []
        tokenize_2t01_ins_index = []
        for i in range(batch_size):
            text1.append(data[i].prompted_sentence1)
            text2.append(data[i].prompted_sentence2)
            tokenize_1t02_ins_index.append(data[i].tokenize_1t02_ins_index)
            tokenize_2t01_ins_index.append(data[i].tokenize_2t01_ins_index)
            # wsy：cause和causedby  翻转
            if data[i].cause1 == 0:
                batch_labels_1.append(0)
                batch_labels_2.append(0)

                batch_labels_5.append(0)
                batch_labels_5.append(0)
            elif data[i].cause1 == 1:
                batch_labels_1.append(1)
                batch_labels_2.append(2)

                batch_labels_5.append(1)
                batch_labels_5.append(1)
            else:
                batch_labels_1.append(2)
                batch_labels_2.append(1)

                batch_labels_5.append(1)
                batch_labels_5.append(1)
        # wsy：truncation=True 超过最大长度进行截断 output1是字典（input_ids，attention_mask...）
        output1 = self.tokenizer(
            text1,
            padding=PaddingStrategy.LONGEST,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids1: torch.Tensor = output1["input_ids"]

        # wsy: mask_pos1 = (input_ids1 == self.tokenizer.mask_token_id).long().argmax(dim=1)
        # torch.nonzero().shape=(batch_size*2,2)
        mask_pos1=torch.nonzero(input_ids1==self.tokenizer.mask_token_id)
        masks1 = output1["attention_mask"]
        input_ids_for_new1 = input_ids1 - self.raw_vacob_size + 1
        # wsy: 将 input_ids_for_new1 中小于 0 的元素替换为 0
        input_ids_for_new1 = torch.where(input_ids_for_new1 < 0, torch.tensor(0), input_ids_for_new1)
        # wsy: 将 input_ids1 中>= 词表长度 的元素替换为 6666
        input_ids1 = torch.where(input_ids1 >= (self.raw_vacob_size), torch.tensor(6666), input_ids1)
        labels1 = torch.tensor(data=batch_labels_1, dtype=torch.long)

        output2 = self.tokenizer(
            text2,
            padding=PaddingStrategy.LONGEST,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids2: torch.Tensor = output2["input_ids"]
        # mask_pos2 = (input_ids2 == self.tokenizer.mask_token_id).long().argmax(dim=1)
        mask_pos2 = torch.nonzero(input_ids2 == self.tokenizer.mask_token_id)
        masks2 = output2["attention_mask"]
        input_ids_for_new2 = input_ids1 - self.raw_vacob_size + 1
        input_ids_for_new2 = torch.where(input_ids_for_new2 < 0, torch.tensor(0), input_ids_for_new2)
        input_ids2 = torch.where(input_ids2 >= (self.raw_vacob_size), torch.tensor(6666), input_ids2)
        labels2 = torch.tensor(data=batch_labels_2, dtype=torch.long)

        labels5 = torch.tensor(data=batch_labels_5, dtype=torch.long)

        # tokenize_1t02_ins_index = torch.tensor(data=tokenize_1t02_ins_index, dtype=torch.long)
        # tokenize_2t01_ins_index = torch.tensor(data=tokenize_2t01_ins_index, dtype=torch.long)
        # wsy：将 data 中的每个元素的 signal 属性提取出来
        # batch_signals = torch.tensor(list(map(lambda x: x.signal, data)), dtype=torch.long)
        return input_ids1, masks1, input_ids_for_new1, mask_pos1, labels1, tokenize_1t02_ins_index, \
            input_ids2, masks2, input_ids_for_new2, mask_pos2, labels2, tokenize_2t01_ins_index, labels5


class BaselineSampler(Sampler):
    def __init__(self, data_source: List[BaselineInputfeature], replacement: bool = True) -> None:
        super().__init__(data_source=data_source)
        positive_index: List[int] = []
        negative_index: List[int] = []
        # wsy: cause1非0即为正例
        for index, inputfeature in enumerate(data_source):
            if inputfeature.cause1:
                positive_index.append(index)
            else:
                negative_index.append(index)
        positive_weight = (len(negative_index) / len(positive_index)) / 7
        weights = torch.ones([len(data_source)], dtype=torch.float32)
        positive_indices = torch.tensor(positive_index, dtype=torch.long)
        # wsy：一维向量，负例索引位置为1，正例索引位置positive_weight
        weights[positive_indices] = positive_weight
        # wsy:cai yang qi
        self.impl = WeightedRandomSampler(weights=weights, num_samples=len(data_source), replacement=replacement)

    def __iter__(self):
        return self.impl.__iter__()

    def __len__(self) -> int:
        return len(self.impl)