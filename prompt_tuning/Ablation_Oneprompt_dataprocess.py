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


"""bert input sequence max length """
MAX_LENTH = 512


def baseline_init_tokenizer(name_or_path: str, save_dir: str) -> Tuple[
    Union[BertModel, RobertaModel], Union[BertTokenizer, RobertaTokenizer]]:
    """初始化分词器,加入8个特殊字符"""

    # mlm:AutoModel=AutoModel.from_pretrained(name_or_path)
    mlm_tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    """分词器增加特殊字符"""
    special_tokens_dict = {"additional_special_tokens": [T1, T2, T3, T4, T5, T6]}
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

    # wsy:将触发词替换为模板形式：</t1> event1 </t2>  </t3> event2 </t4>
    new_tokens = tokens.copy()
    new_tokens[source_start_index] = T1 + " " + new_tokens[source_start_index]
    new_tokens[source_end_index] = new_tokens[source_end_index] + " " + T2
    new_tokens[target_start_index] = T3 + " " + new_tokens[target_start_index]
    new_tokens[target_end_index] = new_tokens[target_end_index] + " " + T4
    # wsy:完整s The event </t1>event1</t2> has the </t5> [Mask] </t6> of the event </t3> event2 </t4>.

    """"
        only using Causal prompt
    """
    prompted_tokens = \
        new_tokens[substr_token_start_index:substr_token_end_index + 1] + \
        ["The event"] + \
        new_tokens[source_start_index:source_end_index + 1] + \
        ["has the", T5, mask_token, T6, "of the event"] + \
        new_tokens[target_start_index:target_end_index + 1] + ["."]
    return " ".join(prompted_tokens)

def baseline_preprocess_data(data: Dict[str, Any], tokenizer: Union[BertTokenizer, RobertaTokenizer]) -> List[
    BaselineInputfeature]:
    temporality=0
    tokens = data["tokens"]
    token_index2sentence_index = data["token_index2sentence_index"]
    sentences = data["sentences"]
    relations: Dict[str, int] = data["relations"]
    res = []
    # esl
    if relations is None:
        return res
    for rel in relations:
        event1_start_index = rel["event1_start_index"]-1
        event1_end_index = rel["event1_end_index"]-1
        event2_start_index = rel["event2_start_index"]-1
        event2_end_index = rel["event2_end_index"]-1

        cause = rel["relation"][0]
        if cause=="cause":
            cause=1


        if cause=="caused":
            cause=-1


        if cause==0:
            cause = 0




        substr_token_start_index = -1000
        substr_token_end_index = -1000

        substr_token_start_index = sentences[token_index2sentence_index[min(event1_start_index, event2_start_index)]]["start"]
        substr_token_end_index = sentences[token_index2sentence_index[max(event1_end_index, event2_end_index)]]["end"]
        if substr_token_end_index - substr_token_start_index > 300:
            continue
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
        # if prompt1==None:
        #     continue

        # wsy:翻转event1和event2
        event1_start_index, event1_end_index, event2_start_index, event2_end_index = event2_start_index, event2_end_index, event1_start_index, event1_end_index
        prompt2 = make_prompted(
            tokens=tokens,
            source_start_index=event1_start_index,
            source_end_index=event1_end_index,
            target_start_index=event2_start_index,
            target_end_index=event2_end_index,
            mask_token=tokenizer.mask_token,
            substr_token_start_index=substr_token_start_index,
            substr_token_end_index=substr_token_end_index
        )

        res.append(BaselineInputfeature(cause, -cause, prompt1, prompt2))
    return res


def valid_data_preprocess(data: Dict[str, Any]):

    for rel in data["relations"]:
        if rel["signal_start_index"] >= 0:
            rel["signal"] = True
        rel["signal_start_index"] = -1
        rel["signal_end_index"] = -1

    return data


class BaselineCollator:
    def __init__(self, tokenizer: Union[BertTokenizer, RobertaTokenizer]) -> None:
        self.tokenizer = tokenizer
        self.raw_vacob_size = self.tokenizer.vocab_size

    def __call__(self, data: List[BaselineInputfeature]) -> Tuple[torch.Tensor, ...]:
        batch_size = len(data)
        text1, text2 = [], []
        batch_labels_1, batch_labels_2 = [], []

        for i in range(batch_size):
            text1.append(data[i].prompted_sentence1)
            text2.append(data[i].prompted_sentence2)
            # wsy：cause和causedby  翻转
            if data[i].cause1 == 0:

                batch_labels_1.append(0)
                batch_labels_2.append(0)

            elif data[i].cause1 == 1:
                batch_labels_1.append(1)
                batch_labels_2.append(2)

            else:
                batch_labels_1.append(2)
                batch_labels_2.append(1)

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
        """
        完整s The event </t1>event1</t2> has the </t5> tokenizer.mask_token </t6> of the event </t3> event2 </t4>.
        wsy：找到输入序列 input_ids1 中的 <mask> 标记的位置
        1.input_ids1==self.tokenizer.mask_token_id：将输入序列 input_ids1 中与 <mask> 标记对应的位置与 mask_token_id 进行比较，得到一个布尔类型的张量，标记了 <mask> 位置为 True，其他位置为 False。
        2.long()：将布尔类型的张量转换为整数类型，其中 True 转换为 1，False 转换为 0。
        3.argmax(dim=1)：在第1维（即序列中的每个位置）上求取最大值的索引。由于 <mask> 标记只会出现一次，因此在整个序列中只有一个位置的值为 1，其他位置的值为 0。因此，通过求取最大值索引，可以找到唯一的 <mask> 标记的位置。
        """
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

        # wsy：将 data 中的每个元素的 signal 属性提取出来
        # batch_signals = torch.tensor(list(map(lambda x: x.signal, data)), dtype=torch.long)
        return \
            input_ids1, masks1, input_ids_for_new1, mask_pos1, labels1,\
                input_ids2, masks2, input_ids_for_new2, mask_pos2, labels2
