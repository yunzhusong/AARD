import os
import json
import pdb
import argparse
import numpy as np
from tqdm import tqdm
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from pytorch_pretrained_bert import BertTokenizer
import preprocessor as pre
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pandas as pd
import random
from collections import defaultdict

cwd = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('-write_data_text',  action='store_true')
    parser.add_argument('-write_data_idx', action='store_true')
    parser.add_argument('-path_raw', type=str, default='../Dataset/pheme')
    parser.add_argument('-data_dir',  type=str, default='data/Pheme')
    parser.add_argument('-text_file',  type=str, default='data.text.txt')
    parser.add_argument('-idx_file',  type=str, default='data.TD_RvNN.vol_5000.txt')
    parser.add_argument('-label_file',type=str, default='data.label.txt')
    parser.add_argument('-split_event', action='store_true')
    
    args = parser.parse_args()
    return args


def get_days(day, month, year):
    days = [0, 31,28,31,30,31,30,31,31,30,31,30,31]
    months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
             'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}

    # 閏年
    if year %4 == 0:
        add1 = True
        days_year = 366
    else:
        add1 = False
        days_year = 365

    # get the number of days before this day
    mon = months[month]
    days_before = sum(days[:mon]) + day-1 # -1 for not acount today
    if mon > 2 and add1:
        days_before += 1

    # get the number of days after the day
    days_after = days_year - days_before
    return days_before, days_after


def get_minutes(clock):
    hour, minute, sec = clock.split(':')
    hour = int(hour)
    minute = int(minute)
    sec = int(sec) / 60
    return hour*60 + minute + sec


def get_time_interval(time1, time2):
    week1, mon1, day1, clock1, zone1, year1 = time1.split()
    week2, mon2, day2, clock2, zone2, year2 = time2.split()
    year1, year2 = int(year1), int(year2)
    day1, day2 = int(day1), int(day2)

    # Should in the same zone
    assert zone1 == zone2
    if year1 == year2:

        days1,_ = get_days(day1, mon1, year1)
        days2,_ = get_days(day2, mon2, year2)
        mins1 = get_minutes(clock1)
        mins2 = get_minutes(clock2)

        min1_tot = days1*24*60 + mins1
        min2_tot = days2*24*60 + mins2

        interval = min2_tot - min1_tot
        if interval < 0:
            pdb.set_trace()
        return interval

    else:
        assert year1 < year2

        _,days1 = get_days(day1, mon1, year1)
        days2,_ = get_days(day2, mon1, year2)
        mins1 = get_minutes(clock1)
        mins2 = get_minutes(clock2)

        min1_tot_left = days1*24*60 - mins1
        min2_tot = days2*24*60 + mins2

        interval = min1_tot_left + min2_tot
        return min1_tot_left + min2_tot



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


def build_loopup_table(freq_list):
    table = {}
    idx = 0
    for item in freq_list:
        table[item[0]] = idx
        idx += 1
    return table

## 2.
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


def max_length(current_max, new_text):
    new_length = len(new_text.split())
    if new_length > current_max:
        return new_length
    return current_max

## 1.
def extract_raw_to_text(args):
    events = ['charliehebdo','ferguson','germanwings-crash','ottawashooting','sydneysiege']
    #num_data={'charliehebdo':458,'ferguson':284,'germanwings-crash':231,'ottawashooting':420,'sydneysiege':522}
    labels = ['rumours', 'non-rumours']
    path_raw = args.path_raw
    path_out = os.path.join(cwd, args.data_dir)

    ## Statistics data
    cnt, cnt_tree = 0, 0
    num_label = {'rumours': 0, 'non-rumours':0}
    text_lengths = []
    tree_sizes = []


    if not os.path.exists(path_out):
        os.makedirs(path_out)

    ## Format label file

    if args.write_data_text:
        #f_label = open(os.path.join(path_out, args.label_file), 'w')
        f_data  = open(os.path.join(path_out, args.text_file), 'w')

    ## Write splited label
    all_rum = []
    all_non = []

    statis = {}
    data_event = defaultdict(list)
    for event in events:
        path_event = os.path.join(path_raw, event)

        statis[event] = {}
        for label in labels:
            path_label = os.path.join(path_event, label)
            rootids = os.listdir(path_label)
            statis[event][label] = 0

            cnt_num = 0
            for rootid in tqdm(rootids):
                path_react = os.path.join(path_label, rootid, 'reactions')
                reacts = os.listdir(path_react)

                # Remove the data without response
                if len(reacts)==0:
                    continue
                num_label[label] += 1


                # Valid num reach the required
                #if cnt_num == num_data[event]:
                #    break
                cnt_num += 1 

                if args.write_data_text:
                    line = '{}\t{}\t{}\n'.format(label, event, rootid)
                    #f_label.write(line)

                if label == 'rumours':
                    all_rum.append(line)
                elif label == 'non-rumours':
                    all_non.append(line)
                data_event[event].append(line)


    if args.split_event:
        with open(os.path.join(path_out, args.label_file), 'w') as f:
            all_ = all_rum + all_non
            f.write(''.join(all_).rstrip())

        for event in events:
            all_events = events.copy()
            all_events.remove(event)

            print(all_events)
            path_split = os.path.join(path_out, 'split_{}'.format(event))
            os.makedirs(path_split, exist_ok=True)

            f_test = open(os.path.join(path_split, 'test.label.txt'), 'w')
            f_train = open(os.path.join(path_split, 'train.label.txt'), 'w')

            test = data_event[event]
            train = []
            for e in all_events:
                train += data_event[e]

            f_test.write(''.join(test).rstrip())
            f_train.write(''.join(train).rstrip())

            f_test.close()
            f_train.close()

    

    else:
        SEED = 777
        random.seed(SEED)

        num_r = len(all_rum)
        num_n = len(all_non)
        random.shuffle(all_rum)
        random.shuffle(all_non)

        cut = int(num_r/5)
        all_rum = all_rum[:cut*5]
        all_non = all_non[:cut*5]
        
        print('Splitting into 5 folds')
        for i in range(5):
            path_split = os.path.join(path_out, 'split_{}'.format(i))
            os.makedirs(path_split, exist_ok=True)

            f_test = open(os.path.join(path_split, 'test.label.txt'), 'w')
            f_train = open(os.path.join(path_split, 'train.label.txt'), 'w')

            test = all_rum[ cut*i : cut*(i+1) ] + all_non[ cut*i : cut*(i+1) ]
            train= all_rum[:cut*i] + all_rum[cut*(i+1):] + all_non[:cut*i] + all_non[cut*(i+1):]

            print('test:', len(test), 'train:', len(train))

            f_test.write(''.join(test).rstrip())
            f_train.write(''.join(train).rstrip())

            f_test.close()
            f_train.close()

        with open(os.path.join(path_out, args.label_file), 'w') as f:
            all_ = test + train
            f.write(''.join(all_).rstrip())

    ## Write text data

    lines = all_rum + all_non
    for line in tqdm(lines):
        label, event, rootid = line.rstrip().split('\t')
        statis[event][label] += 1

        path_event = os.path.join(path_raw, event)

        path_label = os.path.join(path_event, label)
        path_react = os.path.join(path_label, rootid, 'reactions')

        reacts = os.listdir(path_react)

        # Remove the data without response
        assert len(reacts)!=0

        if args.write_data_text:
            line = '{}\t{}\t{}\n'.format(label, event, rootid)
            #f_label.write(line)

        idx = 2
        #num_parent = 0
        max_text_length = 0
        cnt_no_parent = 0
        parent_nodes = []
        id_to_idx = {'root':'None'}
        id_content = {}
        id_interval = {}

        parent_nodes.append(int(rootid)) # root node is a parent node
        parent_nodes.append('root')

        # find all existing node in data folder
        valid_id = [int(i.replace('.json', '')) for i in reacts]
        valid_id.append(int(rootid))

        # Read source data
        path_source = os.path.join(path_label, rootid, 'source-tweet/{}.json'.format(rootid))
        with open(path_source, 'r') as f:
            data = json.load(f)
            time_start = data['created_at']
            text = data['text']
            text = clean_data(text)
            id_content[int(rootid)] = ['root', text, 0]
            id_interval[int(rootid)] = 0
            max_text_length = max_length(max_text_length, text)

        # Read reaction
        for react in reacts:
            with open(os.path.join(path_react, react), 'r') as f_react:
                data = json.load(f_react)
                self_id = data['id'] # 自己的post id
                text = data['text'] # post content
                parent_id = data['in_reply_to_status_id'] # parent id
                time_reply = data['created_at']

                if not (parent_id in parent_nodes): # if exist other parent node
                    parent_nodes.append(parent_id)

                interval = get_time_interval(time_start, time_reply)

                ## Record information by node id
                text = clean_data(text)
                if self_id == int(rootid):
                    continue
                id_content[self_id] = [parent_id, text, interval] # self idx = [parent idx, content, interval]
                id_interval[self_id] = interval
                max_text_length = max_length(max_text_length, text)

        # Order the id by interval and assign the index
        idx = 1
        id_interval = dict(sorted(id_interval.items(), key=lambda item: item[1]))
        for k, v in id_interval.items():
            id_to_idx[k] = idx
            idx += 1

        num_parent = len(set(parent_nodes))
        ## Write to the data file
        id_content = dict(sorted(id_content.items(), key=lambda item: item[1][2]))
        for id_ in id_content.keys():
            #text = clean_data(idx_content[idx][1])
            parent_id, text, interval = id_content[id_]

            # If the parent id doesn't exist, assign the parent to root
            if parent_id in id_to_idx.keys():
                parent_idx = id_to_idx[parent_id]
            else:
                parent_idx = 1
                cnt_no_parent += 1

            self_idx = id_to_idx[id_]
            if parent_idx == 1 and self_idx ==1 :
                pdb.set_trace()
            line = '{}\t{}\t{}\t{}\t{}\t{:.1f}\t{}\n'\
                .format(rootid, parent_idx, self_idx, num_parent, max_text_length, interval, text)
            if args.write_data_text:
                f_data.write(line)

            text_lengths.append(len(text.split()))
            cnt += 1
        cnt_tree += 1
    if args.write_data_text:
        f_data.close()


    with open(os.path.join(path_out, 'README'), 'w') as f:
        f.write('# of tree: {}\n'.format(cnt_tree))
        f.write('# of node: {}\n'.format(cnt))
        f.write('# of rumour: {}\n'.format(len(all_rum)))
        f.write('# of non-rumour: {}\n'.format(len(all_non)))
        f.write('Avg text length: {}\n'.format(sum(text_lengths)/len(text_lengths)))
        f.write('Max text length: {}\n'.format(max(text_lengths)))
        f.write('Min text length: {}\n'.format(min(text_lengths)))
        f.write('# of nodes without parent'.format(cnt_no_parent))
        f.write(str(statis))


    print('# of tree: ', cnt_tree)
    print('# of node: ', cnt)
    print('# of rumour: ', len(all_rum))
    print('# of non-rumour: ', len(all_non))
    print('Avg text length: ', sum(text_lengths)/len(text_lengths))
    print('Max text length: ', max(text_lengths))
    print('Min text length: ', min(text_lengths))
    print('# of nodes without parent'.format(cnt_no_parent))
    print(statis)
    #print('Avg tree size: ', sum(tree_sizes)/len(tree_sizes))


def clean_data(line):

    ## Remove @, reduce length, handle strip
    tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)
    line = ' '.join(tokenizer.tokenize(line))

    ## Remove url, emoji, mention, prserved words, only preserve smiley
    #pre.set_options(pre.OPT.URL, pre.OPT.EMOJI, pre.OPT.MENTION, pre.OPT.RESERVED)
    pre.set_options(pre.OPT.URL, pre.OPT.RESERVED)
    line = pre.tokenize(line)

    ## Remove non-sacii 
    line = ''.join([i if ord(i) else '' for i in line]) # remove non-sacii
    #if not line:
    #    line = 'RT'
    line = line + ' <end>'

    """
    line = line.replace(r'https?://\S+', r'') # remove url
    if line.startswith('RT @'): 
        line = line.replace(r'RT ', r'') # remove RT (retweet)"""
    return line



if __name__=='__main__':
    args = parse_args()

    print(args)
    #if args.write_data_text:
    extract_raw_to_text(args)

    if args.write_data_idx:
        convert_text_to_idx(args)
