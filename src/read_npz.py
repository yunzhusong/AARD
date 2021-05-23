import os
import numpy as np
import pdb

path = '/data1/home2/RumorGD/results/Twitter16/filtergcndetector_gen/0/twitter16_pollute_textgraph'

filename = os.listdir(path)
for file in filename:

    path1 = os.path.join('/home/yunzhu/Workspace/RumorGD/dataset/twitter16textgraph', file)
    path2 = os.path.join('/data1/home2/RumorGD/results/Twitter16/filtergcndetector_gen/0/twitter16_pollute_textgraph', file)

    data1 = np.load(path1, allow_pickle=True)
    data2 = np.load(path2, allow_pickle=True)


    print(data1['x'])
    print(len(data1['x']))
    print(data2['x'])
    print(len(data2['x']))
    pdb.set_trace()
