import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import re
import json
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import GloVe
import spacy
nlp = spacy.load("en_core_web_sm", disable = ["parser", "tagger", "ner"])

class WordEncoder(nn.Module):
    def __init__(self, args, vocab_vectors):
        super(WordEncoder, self).__init__()
        self.emb = nn.Embedding(len(vocab_vectors), args.det_text_emb_dim)
        self.emb.weight.data.copy_(vocab_vectors)
        self.emb.weight.requires_grad = args.det_train_text

    def forward(self, src_seq):
        return self.emb(src_seq)


class BuildVocab(object):
    def __init__(self, args):
        super(BuildVocab, self).__init__()

        self.path = os.path.join(args.dataset_dir, 'data.text.txt')
        self.stoi = None
        self.vectors = None


    def build(self):
        print('Build Vocabulary from ', self.path)

        tokenize = BuildVocab.tokenize_text
        TEXT = Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=35, use_vocab=True)
        datafields = [('eid', None),('idxP',None),('idxC',None),('MaxDegree',None),('MaxL',None),('text', TEXT)]

        data = TabularDataset(path=self.path, format='tsv', skip_header=False, fields=datafields)       
        TEXT.build_vocab(data, vectors=GloVe(name='6B', dim=300), max_size=1000) 

        #train_iter = BucketIterator(train_data, batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)
        self.stoi = TEXT.vocab.stoi
        self.vectors = TEXT.vocab.vectors


    @staticmethod
    def tokenize_text(text):
        text = BuildVocab.clean_text(text)
        token_lst = [token.text.lower() for token in nlp(text)]
        #token_lst = DataLoader.clean_tokenized_text(token_lst)
        return token_lst


    @staticmethod
    def clean_text(text):
        """
        This function cleans the text in the following ways
        1. Replace websites with URL
        2. Replace 's with <space>'s (e.g., her's --> her 's)
        """
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "URL", text) # Replace urls with special token
        text = text.replace("\'s", "")
        text = text.replace("\'", "")
        text = text.replace("n\'t", " n\'t")
        text = text.replace("@", "")
        text = text.replace("#", "")
        text = text.replace("_", " ")
        text = text.replace("-", " ")
        text = text.replace("&amp;", "")
        text = text.replace("&gt;", "")
        text = text.replace("\"", "")
        text = text.replace(".", "")
        text = text.replace(",", "")
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = ' '.join(text.split())
        return text.strip()



def construct_tree_by_Ian(edges):
    forest = {}
    # Populate the forest with depth-1 trees, with placeholders for subtrees.
    for (parent, child) in edges:
        if parent not in forest:
            forest[parent] = {}
        forest[parent][child] = {}
    if not forest:
  # Graph is empty; an edge list cannot represent a single-node tree.
        return {}
    roots = set(forest.keys())
    # Replace the placeholders with corresponding trees from the forest.
    for tree in forest.values():
        roots -= tree.keys()
        for child in tree:
            if child in forest:
                tree[child] = forest[child]
    # Make sure we have a single root.  No check is made for cycles.
    if len(roots) == 0 or len(roots) > 1:
        raise ValueError("Graph is not a tree")
    # The tree located at the actual root contains the entire tree.
    root = roots.pop()
    tree = {root: forest[root]}
    return tree

def construct_tree_from_edges(edges):
    for edge in edges:
        parent, child = edge

def find_depth(idx, edges, depth_vector, current_depth):
    edge_idx = torch.where(edges[0]==idx)[0]
    if len(edge_idx)==0:
        return depth_vector
    node_idx = edges[1, edge_idx]
    depth_vector.scatter_(0, edge_idx, 1+0.8**(current_depth))

    for idx in node_idx:
        depth_vector = find_depth(idx, edges, depth_vector, current_depth+1)
    return depth_vector

def construct_depth_vector(root_idx, edges):
    depth_vector = torch.ones(edges.size(1))
    depth_vector = find_depth(root_idx, edges, depth_vector, 0)
    return depth_vector


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def cal_attacker_loss2(prediction, data_att, mu, logvar, target_label, select_idx):

    target_label = target_label.to(data_att.y.device)
    loss_frame = F.nll_loss(F.log_softmax(prediction, dim=1), target_label)
    mse_label = torch.zeros(prediction.shape).to(data_att.y.device).scatter_(1,data_att.y.reshape(-1,1),1)
    loss_att = -F.mse_loss(torch.softmax(prediction,dim=1), mse_label)
    loss_kl = KL_loss(mu, logvar)
    loss_rec = F.mse_loss(data_att.x[select_idx], data_att.x[-len(select_idx):])
    loss = loss_att  + loss_kl + 2*loss_rec + loss_frame

    _, pred = prediction.max(dim=-1)
    correct = pred.eq(data_att.y).sum().item()
    acc = correct/len(data_att.y)

    correct = pred.eq(target_label).sum().item()
    acc_tar = correct/len(target_label)
    len_gen = int(torch.mean(torch.sum(data_att.x[-len(select_idx):],dim=1)).item()) # record generated length
    print('Length of generated comments:', len_gen)
    record = {
            'loss': loss.item(),
            'loss_frame': loss_frame.item(),
            'loss_att': loss_att.item(),
            'loss_kl': loss_kl.item(),
            'loss_rec': loss_rec.item(),
            'acc': acc,
            'acc_target': acc_tar,
            'len_gen': len_gen,
            }

    return loss, acc, acc_tar, record
def cal_attacker_loss(detector, data_att, mu, logvar, target_label, select_idx):

    prediction = detector(data_att)
    target_label = target_label.to(data_att.y.device)
    loss_frame = F.nll_loss(F.log_softmax(prediction, dim=1), target_label)
    mse_label = torch.zeros(prediction.shape).to(data_att.y.device).scatter_(1,data_att.y.reshape(-1,1),1)
    loss_att = -F.mse_loss(torch.softmax(prediction,dim=1), mse_label)
    loss_kl = KL_loss(mu, logvar)
    loss_rec = F.mse_loss(data_att.x[select_idx], data_att.x[-len(select_idx):])
    loss = loss_att  + loss_kl + 2*loss_rec + loss_frame

    _, pred = prediction.max(dim=-1)
    correct = pred.eq(data_att.y).sum().item()
    acc = correct/len(data_att.y)

    correct = pred.eq(target_label).sum().item()
    acc_tar = correct/len(target_label)
    len_gen = int(torch.mean(torch.sum(data_att.x[-len(select_idx):],dim=1)).item()) # record generated length
    print('Length of generated comments:', len_gen)
    record = {
            'attacker/loss': loss.item(),
            'attacker/loss_frame': loss_frame.item(),
            'attacker/loss_att': loss_att.item(),
            'attacker/loss_kl': loss_kl.item(),
            'attacker/loss_rec': loss_rec.item(),
            'attacker/acc': acc,
            'attacker/acc_target': acc_tar,
            'attacker/len_gen': len_gen,
            }

    return loss, acc, acc_tar, record

def KL_loss(mu, logvar):
    # -0.5 * sum(1+log(sigma^2)-mu^2-sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def compute_identifier_loss(identifier,data_att,  id_label_rec, id_label_gen, N_gen):
    criterion = nn.BCELoss()
    # <- Do we need to average for each data (differernt size of tree)
    prediction = identifier(data_att)
    loss_rec = criterion(prediction[:-N_gen], id_label_rec)
    loss_gen = criterion(prediction[-N_gen:], id_label_gen)
    loss = loss_rec + loss_gen

    acc_rec = ((prediction[:-N_gen]>0.5) == id_label_rec).sum() / float(len(id_label_rec))
    acc_gen = ((prediction[-N_gen:]>0.5) == id_label_gen).sum() / float(len(id_label_gen))
    acc = (acc_rec + acc_gen)/2

    return loss, loss_rec, loss_gen, acc, acc_rec, acc_gen

def compute_attacker_loss(identifier, detector, data, data_att, id_label_rec, id_label_gen, N_gen):
   
    # Reconstruction loss <- L1 loss? L2 loss? regularization?
    mse = nn.MSELoss()
    loss_rec = mse(data_att.x[:-N_gen], data.x)

    rec_list = [loss_rec]

    # Adversarial loss
    bce = nn.BCELoss()

    for param in identifier.parameters():
        param.requires_grad = False
    prediction = identifier(data_att)
    loss_ori = bce(prediction[:-N_gen], id_label_rec)
    loss_gen = bce(prediction[-N_gen:], id_label_gen)
    loss_adv = -(loss_rec + loss_gen)/2 #'-' for Max

    acc_rec = ((prediction[:-N_gen]>0.5) == id_label_rec).sum() / float(len(id_label_rec))
    acc_gen = ((prediction[-N_gen:]>0.5) == id_label_gen).sum() / float(len(id_label_gen))
    acc = (acc_rec + acc_gen)/2

    adv_list = [loss_adv, loss_ori, loss_gen, acc, acc_rec, acc_gen]

    # Reverse opinion loss
    for param in detector.parameters():
        param.requires_grad = False
    #prediction_ori = detector(data)
    prediction_att = detector(data_att)
    # <- Hard or soft label
    # <- Get rid of np.exp
    #prediction_att = torch.exp(prediction_att)
    #loss_rev = F.nll_loss(torch.log((1-prediction_att)), data.y)
    temp = torch.log((1-F.softmax(prediction_att, dim=1)+0.2))
    loss_rev = F.nll_loss(temp, data.y) # '-' for Max
    #loss_rev = - F.nll_loss(prediction_att, prediction_ori) # '-' for Max
    bs = data.y.size(0)
    #fail_rev = (torch.max(prediction_ori, dim=1)[1] == torch.max(prediction_att, dim=1)[1]).sum().float()/bs
    fail_rev = (torch.max(prediction_att, dim=1)[1] == data.y).sum().float()/bs

    rev_list = [loss_rev, fail_rev]

    return rec_list, adv_list, rev_list



def merge_nodes_idxcnt(dataname, savepath):
    from Process.processPheme import text_to_idx_count
    with open('./data/{}/dict.json'.format(dataname)) as f:
        table = json.loads(f.read())

    with open('./data/{}/data.TD_RvNN.vol_5000.txt'.format(dataname)) as f:
        outputlines = f.read().rstrip()
        treeDic = {}
    for line in outputlines.split('\n'):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]) 
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
            
    f_out = open(savepath+'/data.TD_RvNN.vol_5000.txt', 'w')
    f_out.write(outputlines)

    files = os.listdir(savepath+'/temp')
    for file_ in files:
        id_ = file_.split('.')[0]
        new_idx = len(treeDic[id_]) + 1 

        with open(savepath+'/temp/{}.txt'.format(id_)) as f:
            lines = f.read().rstrip().split('\n')

        for i, line in enumerate(lines):
            temp = []
            parent_idx, self_idx, raw_comment = line.split('\t')
            parent_idx = int(parent_idx)
            parent_idx += 1
            idx_count = text_to_idx_count(raw_comment.replace('@ ','@'), table)
            if idx_count == '':
                continue
            temp.append(id_)
            temp.append(str(parent_idx))
            temp.append(str(new_idx))
            temp.append('0')
            temp.append('0')
            temp.append(idx_count)

            new_idx += 1
            new_line = '\t'.join(temp)
            f_out.write('\n{}'.format(new_line))

        f_out.flush()
    f_out.close()


def zero():
    return 0
