import os
import pandas as pd

def build_from_new_input(new_text_file, dataset_dir):
    new_file = '/'.join(new_text_file.split("/")[:-1])
    
    ori_text_file="{}/data.text.txt".format(dataset_dir)
    ori_label_file="{}/data.label.txt".format(dataset_dir)

    for line in open(new_text_file):
        line = line.rstrip()
        eid, indexP, indexC, max_degree, maxL, _, Vec = line.split("\t")
        eid = str(eid)
        indexC = int(indexC)
        max_degree = int(max_degree)
        maxL = int(maxL)

    ori_label = pd.read_csv(ori_label_file, delimiter="\t", header=None)

    lines = []
    for line in open(ori_text_file):
        line = line.rstrip()
        temp = line.split("\t")
        src = temp[-1].replace("\t", " ")
        new_temp = temp[:-1] + [src]
        lines.append(new_temp)

    ori_data = pd.DataFrame(lines)

    chosen_data = ori_data[ori_data[0]==eid]
    chosen_label = ori_label[ori_label[1]==int(eid)]

    exist_nodes = chosen_data[2]
    new_nodes = []
    indexC = len(chosen_data)
    for exist_node in exist_nodes:
        new_nodes.append({0:eid, 1:exist_node, 2:indexC, 3:max_degree, 4:maxL, 5:0, 6:Vec})
        indexC += 1
    new_lines = pd.DataFrame(new_nodes)
    chosen_data = chosen_data.append(new_lines, ignore_index=True)

    return chosen_data, chosen_label

