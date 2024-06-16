from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from typing import Any, Dict, Iterator, List, Optional, Sequence, Sized, Tuple, Union
from dataclasses import asdict, dataclass
import json

if __name__=="__main__":
    with open("C:\wsy\PycharmProjects\PCLearning\data\shortest_path.json", "r") as f:
        datas = json.load(f)
        tokenizer = RobertaTokenizer.from_pretrained("prompt_tuning/mlm")
        # baseline_preprocess_data(datas, tokenizer)
        for data in datas[:1]:
            tok2ori_map = []
            sentence = data["sentence"]
            final_path = data["final_path"]
            alignments_keys = data["alignments_keys"]
            instances_index = data["instances_index"]
            final_path_index = [] # 保存instance 在文本中的索引
            for node in final_path:
                if node in alignments_keys:
                    idx = alignments_keys.index(node)
                    final_path_index.append(instances_index[idx])
                else:
                    print("alignments_keys中没有变量{}".format(node))

            tokens = sentence.split()
            for ori_i, w in enumerate(tokens):
                for t in tokenizer.tokenize(w):
                    tok2ori_map.append(ori_i)
            print(tok2ori_map)
            # 根据final_path_index中的值，统计每个instance所对应的向量
            tokenize_ins_index = []
            for index in final_path_index:
                index_list = [i for i, x in enumerate(tok2ori_map) if x == index]
                tokenize_ins_index.append(index_list)
            print(tokenize_ins_index)


