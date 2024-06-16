import json
import amrlib
import networkx as nx
import penman
from amrlib.alignments.rbw_aligner import RBWAligner
from amrlib.graph_processing.annotator import add_lemmas
from penman import surface
def static_causality(file_name):
    """
    :param file_name:
    :return: count = 1770
    """
    count = 0
    with open(file_name, "r") as f:
        datas = json.load(f)
        for data in datas:
            relations = data["relations"]
            for rel in relations:
                if rel["relation"][0] != 0:
                    if int(rel["event1_sentence_index"]) == int(rel["event2_sentence_index"]):
                        count += 1
    return count
def amr_parsing(file_name):
    """查找最短路径前的准备工作,将必要信息写入JSON文件"""
    with open(file_name, "r") as f:
        stog = amrlib.load_stog_model()
        datas = json.load(f)
        af = open("C:\wsy\PycharmProjects\PCLearning\data\process_esl.json", 'w')
        new_datas = []
        for data in datas:
            print(".....................{}.........................".format(data['filename']))
            relations = data["relations"]
            # esl_file_name = data["filename"]
            # af.write(esl_file_name+'\n')
            tokens = data["tokens"]
            sentences = data["sentences"]
            for rel in relations:
                sentence_start = -1
                sentence_end = -1
                if rel["relation"][0] != 0:
                    if int(rel["event1_sentence_index"]) == int(rel["event2_sentence_index"]):
                        event1_start_index = rel["event1_start_index"]
                        event1_end_index = rel["event1_end_index"]
                        event2_start_index = rel["event2_start_index"]
                        event2_end_index = rel["event2_end_index"]
                        sentence_index = rel["event1_sentence_index"]
                        sentence_start = sentences[sentence_index]["start"] #从0开始
                        sentence_end = sentences[sentence_index]["end"]

                else:
                    event1_start_index = rel["event1_start_index"]
                    event1_end_index = rel["event1_end_index"]
                    event2_start_index = rel["event2_start_index"]
                    event2_end_index = rel["event2_end_index"]
                    for sen in sentences:
                        # event1和event2在同一句话内
                        if event1_start_index-1>=sen["start"] and event1_end_index-1<=sen["end"] and event2_start_index-1>=sen["start"] and event2_end_index-1<=sen["end"]:
                            sentence_start = sen["start"]  # 从0开始
                            sentence_end = sen["end"]

                sen_tokens = tokens[sentence_start:sentence_end+1]
                if len(sen_tokens) == 0:
                    print("rel:{}".format(rel))
                    continue
                sen = " ".join(sen_tokens)
                # af.write(sen)
                # af.write("\n")
                # 重定位2个触发词的位置：event1_start_index索引是从1开始的，而sentences是从0开始的，每个句子从0开始
                e1_start = event1_start_index - 1 - sentence_start
                e1_end = event1_end_index - 1 - sentence_start
                e1 = " ".join(tokens[event1_start_index - 1:event1_end_index])
                e2_start = event2_start_index - 1 - sentence_start
                e2_end = event2_end_index - 1 - sentence_start
                e2 = " ".join(tokens[event2_start_index - 1:event2_end_index])
                # af.write("event1:{0} {1} {2}\n".format(e1, e1_start, e1_end))
                # af.write("event2:{0} {1} {2}\n".format(e2, e2_start, e2_end))
                if rel["relation"][0] == "cause":
                    # af.write("relation:{}\n".format(1))
                    r = 1
                elif rel["relation"][0] == "caused":
                    # af.write("relation:{}\n".format(-1))
                    r = -1
                else:
                    # af.write("relation:{}\n".format(0))
                    r = 0
                try:
                    # AMR解析
                    graphs = stog.parse_sents([sen])
                    # print(graphs)
                except Exception as e:
                    print(f"An error occurred:{e}")
                    print(sen)
                penman_graph = add_lemmas(graphs[0], snt_key='snt')

                aligner = RBWAligner.from_penman_w_json(penman_graph)
                graph_string = aligner.get_graph_string()
                aligned_graph = penman.decode(graph_string)  # get the aligned graph string
                # 聚合对齐的文本
                alignments = surface.alignments(aligned_graph)
                # print(graph_string)
                alignments_keys = list(alignments.keys())
                instances_index = []

                vars = []
                vals = []
                for key in alignments_keys:
                    instances_index.append(alignments[key].indices[0])
                    vars.append(key[0])
                    val = key[2]
                    val = val.strip('"')
                    vals.append(val)


                instances = {}

                for source, role, target in penman_graph.instances():
                    instances[source] = target
                # af.write("penman_graph:{}\n".format(instances))
                # af.write("alignments:{}\n".format(check_vars))
                # af.write("instances_index:{}\n".format(alignments_values))
                new_datas.append({"sentence":sen,\
                                  "event1":e1, "event1_start":e1_start, "event1_end":e1_end,\
                                  "event2":e2, "event2_start":e2_start, "event2_end":e2_end,\
                                  "relation":r,\
                                  "penman_graph_instances":instances,\
                                  "penman_graph_edges":penman_graph.edges(),\
                                  "alignments_keys":vars, \
                                  "alignments_values": vals, \
                                  "instances_index":instances_index})

        json.dump(new_datas, indent=4, fp=af)
        af.close()


from pyvis.network import Network

def shortest_path(file_name):
    G = nx.DiGraph()

    # 创建一个PyVis Network实例
    net = Network(height="800px", width="100%", directed=True)

    jf = open("C:\wsy\PycharmProjects\PCLearning\data\shortest_path.json", "w")
    jf_new_datas=[]
    count_shortest_path1t02 = 0
    count_shortest_path2t01 = 0
    count_error = 0
    with open(file_name, "r") as f:
        datas = json.load(f)
        for data in datas:
            Flag = False
            Flag2t01 = False
            shortest_path_1t02 = []
            shortest_path_2t01 = []
            final_path_1t02 = []
            final_path_2t01 = []

            penman_graph_instances = data["penman_graph_instances"]
            penman_graph_edges = data["penman_graph_edges"]
            instance_sources = penman_graph_instances.keys()
            """构建图"""
            # 添加节点
            for source in instance_sources:
                target = penman_graph_instances[source]
                G.add_node(source, label=target)
            # 添加边
            for source, role, target in penman_graph_edges:
                G.add_edge(source, target, lable=role)

            event1_start = data["event1_start"]
            event2_start = data["event2_start"]
            instances_index = data["instances_index"]

            alignments_keys = data["alignments_keys"]

            if event1_start in instances_index and event2_start in instances_index:
                Aligner_error = "no error"

                # 保证event2和event2都在alignments中
                event1_index = instances_index.index(event1_start)
                event2_index = instances_index.index(event2_start)
                event1_var = alignments_keys[event1_index]
                event2_var = alignments_keys[event2_index]
                # 使用BFS算法找到1to2最短路径
                try:
                    # 尝试查找节点 1 到节点 3 的最短路径
                    shortest_path_1t02 = nx.shortest_path(G, source=event1_var, target=event2_var)
                except nx.NetworkXNoPath:
                    print("没有找到 1to2 路径")
                    Flag = True
                    shortest_path_1t02=[]
                # shortest_path为空
                if Flag:
                    count_shortest_path1t02+=1
                else:
                    # 反查最短路径中的变量是否都有对齐instance
                    for var in shortest_path_1t02:
                        if var in alignments_keys:
                            final_path_1t02.append(var)

                # 使用BFS算法找到 2to1 最短路径
                try:
                    # 尝试查找节点 1 到节点 3 的最短路径
                    shortest_path_2t01 = nx.shortest_path(G, source=event2_var, target=event1_var)
                except nx.NetworkXNoPath:
                    print("没有找到 2to1 路径")
                    Flag2t01 = True
                    shortest_path_2t01 = []
                # shortest_path为空
                if Flag2t01:
                    count_shortest_path2t01 += 1
                else:
                    # 反查最短路径中的变量是否都有对齐instance
                    for var in shortest_path_2t01:
                        if var in alignments_keys:
                            final_path_2t01.append(var)

            else:
                # 如果有一个event不在alignments中，说明AMR工具有误
                count_error+=1
                if event1_start not in instances_index and event2_start not in instances_index:
                    Aligner_error = "event1 and event2 all error"
                elif event1_start not in instances_index:
                    Aligner_error = "event1 error"
                else:
                    Aligner_error = "event2 error"

            # data.append({"shortest_path":shortest_path, "final_path":final_path, "Aligner_error":Aligner_error})
            data["shortest_path_1t02"] = shortest_path_1t02
            data["final_path_1t02"] = final_path_1t02

            data["shortest_path_2t01"] = shortest_path_2t01
            data["final_path_2t01"] = final_path_2t01

            data["Aligner_error"] = Aligner_error
            jf_new_datas.append(data)
        json.dump(jf_new_datas, indent=4, fp=jf)
        jf.close()
    print("1t02 最短路径为空个数：{0}，2t01 最短路径为空个数：{1}，AMR工具解析错误个数：{2}。".format(count_shortest_path1t02, count_shortest_path2t01, count_error))


if __name__=="__main__":
    file_name = "C:\wsy\PycharmProjects\PCLearning\data\esl.json"
    # print(static_causality(file_name))
    # amr_parsing(file_name)
    # file_name2 = "C:\wsy\PycharmProjects\PCLearning\data\process_esl.json"
    # shortest_path(file_name2)