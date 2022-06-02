""" For support load data functions in data/process.py
"""
import os
import pdb
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import GloVe

import re
from data.utils import construct_depth_vector, zero
import spacy
nlp = spacy.load("en_core_web_sm", disable = ["parser", "tagger", "ner"])

def collate_fn(data):
    return data

class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=1, upper=100000, droprate=0, data_path=''):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

class BiGraphTextDataset(Dataset):
    def __init__(self, fold_x, treeDic, tokenizer, lower=1, upper=100000, tddroprate=0,budroprate=0, data_path=''):
        """
        BiGraphTextDataset is to process the raw text data, 
        and convert the raw text into the idx
        Be careful to the dropout control
        """
        if fold_x is not None:
            self.fold_x = list(filter(lambda id: id in treeDic 
                                      and len(treeDic[id]) >= lower 
                                      and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        
        # For process text data
        self.tokenizer = tokenizer
        # Record the class number

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        ## Only few
        #print('Only root node')
        #edgeindex = np.array([[0],[0]])

        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        num_edge = len(new_edgeindex[0])
        depth = torch.ones(len(new_edgeindex[0]))

        x = self.stoi(data['x'])
        root = x[data['rootindex']:data['rootindex']+1] # to remain the dimension
        return Data(x=torch.tensor(x,dtype=torch.long),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(root), 
             rootindex=torch.LongTensor([int(data['rootindex'])]),
             depth=depth, id=torch.LongTensor([int(id)]) , num_edge=torch.ShortTensor([num_edge]))

    def stoi(self, sents):
        sents_idx = np.zeros((len(sents), 30))
        for i in range(len(sents)):
            #tokens = BiGraphTextDataset.clean_text(sents[i]).split()
            tokens = self.tokenizer.tokenize(BiGraphTextDataset.clean_text(sents[i]))
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            sent_len = min(len(ids), 30)
            sents_idx[i][:sent_len] = ids[:sent_len]

        return sents_idx

   
    def get_samples_weight(self):
        """ Get the sample weight for each data by the class number"""
        targets = []
        for i in range(self.__len__()):
            id =self.fold_x[i]
            data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
            targets.append(int(data['y']))
        targets = np.array(targets)
        class_sample_count = np.array(
            [len(np.where(targets==t)[0]) for t in np.unique(targets)])
        weight = 1. /class_sample_count
        samples_weight = np.array([weight[t] for t in targets])
        samples_weight = torch.from_numpy(samples_weight).double()
        return samples_weight

    # Step 1: Define the data fields
    def vocab_builder(self):
        #self.eid_field = Field(sequential=False,tokenize)

        print('Build Vocabulary')
        tokenize = BiGraphTextDataset.tokenize_text
        TEXT = Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=35, use_vocab=True)

        datafields = [('eid', None),('idxP',None),('idxC',None),('MaxDegree',None),('MaxL',None),('text', TEXT)]
        path = '/data1/home2/AgainstRumor/data/Pheme/data.text.txt'
        train_data = TabularDataset(path=path, format='tsv', skip_header=False, fields=datafields) 
        TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300)) 

        #train_iter = BucketIterator(train_data, batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)
        self.stoi_dict = TEXT.vocab.stoi
        self.vocab_vectors = TEXT.vocab.vectors
    

    @staticmethod
    def clean_text(text):
        """
        This function cleans the text in the following ways
        1. Replace websites with URL
        2. Replace 's with <space>'s (e.g., her's --> her 's)
        """
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "URL", text) # Replace urls with special token
        #text = text.replace("\'s", "")
        #text = text.replace("\'", "")
        #text = text.replace("n\'t", " n\'t")
        #text = text.replace("@", "")
        #text = text.replace("#", "")
        #text = text.replace("_", " ")
        #text = text.replace("-", " ")
        text = text.replace("&amp;", "")
        text = text.replace("&gt;", "")
        text = text.replace("\"", "")
        text = text.replace("$MENTION$", '')
        text = text.replace("$ URL $", '')
        text = text.replace("$URL$", '')
        #text = text.replace(".", "")
        #text = text.replace(",", "")
        #text = text.replace("(", "")
        #text = text.replace(")", "")
        text = text.replace("<end>", "")
        text = ' '.join(text.split())
        return text.strip()


    @staticmethod
    def tokenize_text(text):
        text = BiGraphTextDataset.clean_text(text)
        token_lst = [token.text.lower() for token in nlp(text)]
        #token_lst = DataLoader.clean_tokenized_text(token_lst)
        return token_lst

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0, data_path=''):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        num_edge = len(new_edgeindex[0])
        #depth = construct_depth_vector(torch.LongTensor([int(data['rootindex'])]), torch.LongTensor(new_edgeindex))
        depth = torch.ones(len(new_edgeindex[0]))
        
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']), 
             num_edge=torch.LongTensor([num_edge]), rootindex=torch.LongTensor([int(data['rootindex'])]),
             depth=depth)

class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0, data_path=''):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
