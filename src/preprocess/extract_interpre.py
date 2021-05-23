import os
import numpy as np
import json
import pdb
import argparse
from tqdm import tqdm
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from pytorch_pretrained_bert import BertTokenizer
import preprocessor as pre
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

cwd = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('-write_data_text',  action='store_true')
    parser.add_argument('-write_data_idx', action='store_true')
    parser.add_argument('-path_raw', type=str, default='../Dataset/pheme_3class')
    parser.add_argument('-data_dir',  type=str, default='data/Pheme')
    parser.add_argument('-text_file',  type=str, default='data.text.txt')
    parser.add_argument('-idx_file',  type=str, default='data.TD_RvNN.vol_5000.txt')
    parser.add_argument('-label_file',type=str, default='data.label.txt')
    parser.add_argument('-split_event', action='store_true')
    
    args = parser.parse_args()
    return args


def text_to_idx_count(text, table):
    keys = table.keys()
    counts = {}
    for i in text.split():
        if i in keys:
            idx = table[i]
            if idx not in counts.keys():
                counts[idx] = 1
            else:
                counts[idx] += 1
    line = []
    for item in counts.items():
        line.append('{}:{}'.format(item[0], item[1]))
    return ' '.join(line)


def convert_text_to_idx(args):

    path_text = os.path.join(cwd, args.data_dir, args.text_file)
    path_out  = os.path.join(cwd, args.data_dir, args.idx_file)
    if args.write_data_idx:
        f_out = open(path_out, 'w')
    
    ## --------------------------------------- ##
    ## -- Tokenize by BERT-- ##
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data = []
    with open(path_text) as f:
        lines = f.read().split('\n')
    aa = lines[2].split('\t')[-1]
    pdb.set_trace()
    data = [tokenizer.tokenize(line) for line in lines]
    pdb.set_trace()
    """

    ## --Tokenize by torchtext-- ##
    """
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=30)
    datafields = [('eid', None),('idxP',None),('idxC',None),('MaxDegree',None),('MaxL',None),('text', TEXT)]
    train_data = data.TabularDataset(path=path_text, format='tsv', skip_header=False, fields=datafields)
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    table = build_loopup_table(TEXT.vocab.freqs.most_common(5000))
    """
    ## --Build table by idf-- ##
    tokenize = lambda x: x
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=30)
    
    datafields = [('eid', None),('idxP',None),('idxC',None),('MaxDegree',None),('MaxL',None),('interval',None),('text', TEXT)]
    train_data = data.TabularDataset(path=path_text, format='tsv', skip_header=False, fields=datafields)

    corpus = []
    for dd in train_data:
        dd_text = dd.text
        corpus.append(dd_text)

    vectorizer = TfidfVectorizer(token_pattern=r'\S+')
    X = vectorizer.fit_transform(corpus)
    indices = np.argsort(vectorizer.idf_) # sort from large to small by IDF
    feature_names = vectorizer.get_feature_names()
    top_n = 5000 # Find the top n words by IDF
    top_features = [feature_names[i] for i in indices[:top_n]]
    table = {}
    idx = 0
    for feature in top_features:
        table[feature] = idx
        idx += 1
    ## ======================================== ##
    #with open(os.path.join(cwd, 'data/{}/dict_0.json'.format(args.data_dir.split('/')[-1])), 'w') as f_dict:
    #    f_dict.write(json.dumps(table)) # Save the vocabulary dictionary
    with open(path_text, 'r') as f:
        raw_lines = f.read().rstrip().split('\n')

    print('Writing idx:count data file')
    cnt = 0
    for line in tqdm(raw_lines):
        text = line.split('\t')[-1]
        idx_count = text_to_idx_count(text, table)
        temp = line.split('\t')[:-1]
        temp.append(idx_count)
        new_line = '\t'.join(temp)
        cnt += 1
        if args.write_data_idx:
            if cnt == len(raw_lines):
                f_out.write('{}'.format(new_line))
            else:
                f_out.write('{}\n'.format(new_line))

    if args.write_data_idx:
        f_out.close()


def clean_data(tweets):

    outputs = []
    tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)
    for tweet in tweets:
        tweet = ' '.join(tokenizer.tokenize(tweet))
        pre.set_options(pre.OPT.URL, pre.OPT.RESERVED)
        tweet = pre.tokenize(tweet)

        tweet = ''.join([i if ord(i) else '' for i in tweet])
        tweet = tweet + ' <end>'

        outputs.append(tweet)
    return outputs


def get_edge_from_dict(dict_, root, parent_id={}):
    keys = dict_.keys()

    for key in keys:
        parent_id[key] = root
        if dict_[key] == []:
            continue
        else:
            parent_id = get_edge_from_dict(dict_[key], key, parent_id)
    return parent_id

def zero():
    return 0

def extract_raw_to_text(args):

    raw_data_path = args.path_raw
    path_out = args.data_dir

    raw_dir = os.path.join(raw_data_path, 'compiled-data.json')

    with open(raw_dir, 'r') as f:
        raw_data = json.load(f)

    os.makedirs(path_out, exist_ok=True)

    f_data = open(os.path.join(path_out, args.text_file), 'w')

    events = raw_data.keys()
    all_ = []
    line_dict = defaultdict(list)
    cnt_wrong = 0
    statis = defaultdict(list)
    for event in tqdm(events):
        data_event = raw_data[event] # list of datum
       
        statis[event] = defaultdict(zero)
        for datum in data_event:

            if len(datum['tweets'])==0:
                continue

            # Record label
            rootid = datum['id_']
            label = datum['label']

            line = '{}\t{}\t{}\n'.format(label, event, rootid)
            line_dict[event].append(line)
            all_.append(line)
            statis[event][label] += 1

            ids = datum['tweet_ids']
            
            # Obtain idx
            id_to_idx = {}
            id_to_idx[str(rootid)] = 1 # Parent idx is 1
            for n, id_ in enumerate(ids):
                id_to_idx[id_] = n+2 # Count from 2

            # Obtain parent idx
            structure = datum['structure']
            parent_id = get_edge_from_dict(structure, rootid) # {id: parent_id}
            num_parent = len(set(parent_id.values()))

            # Clean text
            tweets = clean_data(datum['tweets'])
            max_text_len = max([len(aa.split()) for aa in tweets])

            claim = clean_data([datum['claim']['tweet']])[0]
            line = '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                rootid, 'None', 1, num_parent, max_text_len, 0, claim)
            f_data.write(line)
            
            for n, id_ in enumerate(ids):
                self_idx = id_to_idx[id_]
                try:
                    parent_idx = id_to_idx[parent_id[id_]]
                except:
                    cnt_wrong += 1
                    #print('{} {} connect to root'.format(event, id_))
                    #pdb.set_trace()
                    parent_idx = 1
                if parent_idx == self_idx:
                    parsent_idx = 'None'
                interval = int(datum['time_delay'][n])
                text = tweets[n]
                
                line = '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    rootid, parent_idx, self_idx, num_parent, max_text_len, interval, text)

                f_data.write(line)

    f_data.close()
    print(cnt_wrong)

    # Write out label txt
    with open(os.path.join(path_out, 'data.label.txt'), 'w') as f:
        f.write(''.join(all_))


    # Split data by event
    for event in tqdm(events):
        out_dir = os.path.join(path_out, 'split_{}'.format(event))
        os.makedirs(out_dir , exist_ok=True)

        f_test = open(os.path.join(out_dir, 'test.label.txt'), 'w')
        f_train = open(os.path.join(out_dir, 'train.label.txt'), 'w')

        train_event = list(events).copy()
        train_event.remove(event)
        test = line_dict[event]
        train = []
        for e in train_event:
            train += line_dict[e]

        f_test.write(''.join(test).rstrip())
        f_train.write(''.join(train).rstrip())

        f_test.close()
        f_train.close()

    print(statis)
    with open(os.path.join(path_out, 'README'), 'w') as f:
        f.write(str(statis))


if __name__=='__main__':
    args = parse_args()

    extract_raw_to_text(args)
    convert_text_to_idx(args)
