# -*- coding: utf-8 -*-
import os
import pdb
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import argparse
from nltk.tokenize import TweetTokenizer
import preprocessor as pre

cwd=os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description='build data graph to npz file')
    parser.add_argument('-format', type=str, default='txt_emb')
    parser.add_argument('-obj', type=str, default='')
    parser.add_argument('-early', type=str, default='')
    parser.add_argument('-class_num', type=int, default=2)
    args = parser.parse_args()
    return args

def clean_data(line):
    ## Remove @, reduce length, handle strip
    tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)
    line = ' '.join(tokenizer.tokenize(line))

    ## Remove url, emoji, mention, prserved words, only preserve smiley
    #pre.set_options(pre.OPT.URL, pre.OPT.EMOJI, pre.OPT.MENTION, pre.OPT.RESERVED)
    #pre.set_options(pre.OPT.URL, pre.OPT.RESERVED, pre.OPT.MENTION)
    pre.set_options(pre.OPT.URL, pre.OPT.RESERVED)
    line = pre.tokenize(line)

    ## Remove non-sacii 
    line = ''.join([i if ord(i) else '' for i in line]) # remove non-sacii
    return line



class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word
    rootfeat = np.zeros([1, 5000])
    if len(root_index)>0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_word=[]
    x_index=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i+1].word)
        x_index.append(index2node[index_i+1].index)
    edgematrix=[row,col]
    return x_word, x_index, edgematrix,rootfeat,rootindex

def constructMat_txt(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        text = tree[j]['vec'] # raw text
        nodeC.text = text
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex=indexC-1
            root_text = text
    matrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_text=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
        x_text.append(clean_data(index2node[index_i+1].text))
    if row == [] and col == []:
        matrix[0][0] = 1
        row.append(0)
        col.append(0)
    edgematrix=[row,col]
    return x_text, edgematrix, root_text, rootindex

def getfeature(x_word,x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x

def buildgraph(obj, format_, class_num, portion=''):
    if format_ == 'idx_cnt':
        treePath = os.path.join(cwd, '../dataset/' + obj + '/data.TD_RvNN.vol_5000.{}txt'.format(portion))
        print("loading ", treePath)
    elif format_ == 'txt_emb':
        treePath = os.path.join(cwd, '../dataset/' + obj + '/data.text.{}txt'.format(portion))
        print("loading ", treePath)
    
    treeDic = {}
    f_tree = open(treePath, 'r')
    for line in f_tree:
        # maxL: max # of clildren nodes for a node; max_degree: max # of the tree depth
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[-1]

        if not treeDic.__contains__(eid):
            # If the event id hasn't been contained
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    f_tree.close()
    print('tree number:', len(treeDic))

    labelPath = os.path.join(cwd, "../dataset/" + obj + "/data.label.txt")

    if class_num == 2:
        labelset_0, labelset_1 = ['non-rumours', 'non-rumor', 'true'], ['rumours', 'rumor', 'false']
        labelset_2, labelset_3 = [], []
    elif class_num == 4:
        labelset_0, labelset_1, labelset_2, labelset_3 = ['true'], ['false'], ['unverified'], ['non-rumor']
    elif class_num == 3:
        labelset_0, labelset_1, labelset_2, labelset_3 = ['true'], ['false'], ['unverified'], []

    print("loading tree label")
    event, y = [], []
    l0 = l1 = l2 = l3 = 0
    labelDic = {}

    f_label = open(labelPath, "r")
    for line in open(labelPath):
        line = line.rstrip()

        label, eid = line.split('\t')[0], line.split('\t')[-1]
        label=label.lower()
        event.append(eid)
        if label in labelset_0:
            labelDic[eid]=0
            l0 += 1
        if label in labelset_1:
            labelDic[eid]=1
            l1 += 1
        if label in labelset_2:
            labelDic[eid]=2
            l2 += 1
        if label in labelset_3:
            labelDic[eid]=3
            l3 += 1
    f_label.close()
    print(len(labelDic))
    print(l1, l2)

    if format_ =='idx_cnt':
        os.makedirs(os.path.join(cwd, '../dataset/'+obj+'graph'), exist_ok=True)
    elif format_ =='txt_emb':
        print(os.path.join(cwd, '../dataset/'+obj+ 'textgraph'))
        os.makedirs(os.path.join(cwd, '../dataset/'+obj+ 'textgraph'), exist_ok=True)

    def loadEid(event, id, y, format_):
        if event is None:
            return None
        if len(event) < 1:
            return None
        if len(event)>= 1:
            if format_ == 'idx_cnt':
                x_word, x_index, tree, rootfeat, rootindex = constructMat(event) 
                x_x = getfeature(x_word, x_index) # x_word: the occur times of words, x_index: the index of words
                rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y)
                np.savez( os.path.join(cwd, '../dataset/'+obj+'graph/'+id+'.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y)
            elif format_ == 'txt_emb':
                x_text, tree, root_text, rootindex = constructMat_txt(event) 
                tree, rootindex, y = np.array(tree), np.array(rootindex), np.array(y)
                np.savez( os.path.join(cwd, '../dataset/'+obj+'textgraph/'+id+'.npz'), x=x_text,root=root_text,edgeindex=tree,rootindex=rootindex,y=y)
            return None

    print("loading dataset", )
    Parallel(n_jobs=1, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid], format_) for eid in tqdm(event))
    return treeDic

if __name__ == '__main__':
    args = parse_args()
    obj = args.obj
    if args.format=='idx_cnt':
        path = os.path.join(cwd, '../dataset/'+obj+'graph/')
        print('Building the graph data by index:cnt at ', path)
    elif args.format=='txt_emb':
        path = os.path.join(cwd, '../dataset/'+obj+'text'+'graph/')
        print('Building the graph data by raw text at ', path)

    portion = '{}.'.format(args.portion)
    os.makedirs(path, exist_ok=True)
    buildgraph(args.obj, args.format, args.class_num, portion)

