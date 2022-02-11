import os
import pdb
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import BatchNorm1d 
torch.multiprocessing.set_sharing_strategy('file_system')
from torchtext import data
from torchtext.legacy.data import Field
from torchtext.legacy.data import Dataset, Example, Iterator, Batch
from torch.distributions.normal import Normal
from collections import defaultdict
from others.logging import logger

cwd = './'


class CommentDataset(Dataset):
    def __init__(self, args, loader, dataname, device, tokenizer=None, tgt_str=False, reverse_label=False):
        """
        Form the dataset (a node) for comment generator from detector's dataset (a tree)
        """
        self.args = args
        cls_token = '[CLS]'
        sep_token = '[SEP]'
        pad_token = '[PAD]'
        tgt_bos = '[unused0]'
        tgt_eos = '[unused1]'
        tgt_sent_split = '[unused2]'
        self.cls_vid = torch.tensor([tokenizer.vocab[cls_token]]) # 101
        self.sep_vid = torch.tensor([tokenizer.vocab[sep_token]]) # 102
        self.pad_vid = torch.tensor([tokenizer.vocab[pad_token]]) # 0
        self.tgt_bos = torch.tensor([tokenizer.vocab[tgt_bos]]) # 1
        self.tgt_eos = torch.tensor([tokenizer.vocab[tgt_eos]]) # 2

        def one():
            return 1
        self.class_dict = defaultdict(one)

        examples = []

        def postprocess(arr, no_use=None):
            return arr

        # data fields
        field = MyField(sequential=True, use_vocab=False, batch_first=True)
        passfield = PassField(sequential=False, use_vocab=False)
        if args.train_gen or args.train_adv:
            fields = [("src",field),("segs",field),("mask_src",field),("label",field),("edges",passfield),("node_batch",passfield),("id",field),("tgt",field),("mask_tgt",field)]
        else:
            fields = [("src",field),("segs",field),("mask_src",field),("label",field),("edges",passfield),("node_batch",passfield),("id",field)]

        #if args.test_gen or tgt_str:
        #    fields.append(("tgt_str", 'string'))

        max_len = self.args.max_length
        max_len_gen = self.args.max_length_gen
        # Iterate every tree (batchsize should be 1)
        for i, tree in enumerate(loader):
            print('Loading Comment Dataset {}\r'.format(i), end='')

            rootindex = tree.rootindex
            source_x = tree.x[rootindex[0]].unsqueeze(0)
            edges = tree.edge_index
            edges = self.correct_edge(edges, rootindex)
            node_batch = tree.batch
            nodes = tree.x[1:]
            label = tree.y
            if reverse_label:
                label = abs(label-1)

            id_ = tree['id']

            # Create sequential data by comment time series
            for n in range(len(nodes)):
                comments_x = nodes[:n] # Take at most 10 nodes
                target_x = nodes[n]

                if len(comments_x) < 3:
                    continue

                if len(comments_x) != 0:
                    src_sents = torch.cat((source_x, comments_x), dim=0)
                else:
                    src_sents = source_x

                src = self.append_and_pad(src_sents, 'src')[:max_len]
                segs = self.find_seg_emb(src_sents)[:max_len]
                tgt = self.append_and_pad(target_x, 'tgt')[:max_len_gen]

                mask_src = src!=self.pad_vid
                mask_tgt = tgt!=self.pad_vid
                #id_idx = [id_, parent_idx, self_idx]

                # Only preserve the edges that in the valid sents
                num_valid_sent = (src==self.cls_vid).sum()
                new_edges = edges[:,edges[1]<num_valid_sent]
                new_node_batch = node_batch[:num_valid_sent] 
                if args.train_gen or args.train_adv:
                    list_ = [src, segs, mask_src, label, new_edges, new_node_batch, id_, tgt, mask_tgt]
                else:
                    list_ = [src, segs, mask_src, label, new_edges, new_node_batch, id_]

                #if args.test_gen or tgt_str:
                #    tgt_txt = ' '.join(tokenizer.convert_ids_to_tokens(target_x[target_x!=self.pad_vid])).replace(' ##', '')
                #    list_.append(tgt_txt)
                self.class_dict[label[0].item()] += 1
                examples.append(MyExample.fromlist(list_, fields))
        if reverse_label:
            logger.info('Finished Loading Comment Dataset with Reverse Label')
        else:
            logger.info('Finish Loading Comment Dataset')

        logger.info(self.class_dict.items())
        super(CommentDataset, self).__init__(examples, fields)

    def append_and_pad(self, idx, mode):
        if mode == 'src':
            cnt_len = 0
            sents = []
            for i, x in enumerate(idx):
                x = x[x!=self.pad_vid]
                x = x[:38]
                sent = torch.cat((self.cls_vid, x, self.sep_vid))
                cnt_len += len(sent)
                sents.append(sent)
            # padding and truncating
            pad_len = max(self.args.max_length-cnt_len,0)
            output = F.pad(torch.cat(sents), (0, pad_len), value=self.pad_vid.item())
            output = output
            return output
                
        elif mode == 'tgt':
            x = idx[idx!=self.pad_vid][:38]
            sent = torch.cat((self.tgt_bos, x, self.tgt_eos))
            # padding
            pad_length = max(self.args.max_length_gen-len(sent), 0)
            output = F.pad(sent, (0,pad_length), value=self.pad_vid.item())
            output = output
            return output

    def find_seg_emb(self, src_sents):
        flag = 0
        cnt_len = 0
        embs = []
        for sent in src_sents:
            sent = sent[sent!=self.pad_vid]
            if flag%2==0:
                pos_emb = torch.zeros(len(sent)+2).long()
            else:
                pos_emb = torch.zeros(len(sent)+2).long()+1
            cnt_len += (len(sent)+2)
            flag += 1
            embs.append(pos_emb)
        # padding
        flag -= 1
        if flag%2==0:
            pos_emb = torch.zeros(max(self.args.max_length-cnt_len,0)).long()
        else:
            pos_emb = torch.zeros(max(self.args.max_length-cnt_len,0)).long()+1
        embs.append(pos_emb)
        segs = torch.cat(embs)
        return segs

    def correct_edge(self, edges, rootindex):
        parents  = edges[0]
        children = edges[1]
        parents[parents>children] = rootindex
        return torch.stack((parents, children), dim=0)



class TreeDataset(Dataset):
    def __init__(self, args, loader, dataname, device, tokenizer, tgt_str=False):
        """
        Form the dataset (a node) for comment generator from detector's dataset (a tree)
        """
        self.args = args
        sep_token = '[SEP]'
        cls_token = '[CLS]'
        pad_token = '[PAD]'
        tgt_bos = '[unused0]'
        tgt_eos = '[unused1]'
        tgt_sent_split = '[unused2]'
        self.sep_vid = torch.tensor([tokenizer.vocab[sep_token]])
        self.cls_vid = torch.tensor([tokenizer.vocab[cls_token]])
        self.pad_vid = torch.tensor([tokenizer.vocab[pad_token]])
        self.tgt_bos = torch.tensor([tokenizer.vocab[tgt_bos]])
        self.tgt_eos = torch.tensor([tokenizer.vocab[tgt_eos]])

        examples = []

        # data fields
        field = MyField(sequential=True, use_vocab=False, batch_first=True)
        passfield = PassField(sequential=True, use_vocab=False, batch_first=True)
        #float_field = data.Field(sequential=True, use_vocab=False, batch_first=True, dtype=torch.float)
        #long_field = data.Field(sequential=True, use_vocab=False, batch_first=True, dtype=torch.long)

        fields = [("src",field),("segs",field),("mask_src",field),("label",field),("edges",passfield),("node_batch",passfield),("id",field)]

        def zero():
            return 0
        self.class_dict = defaultdict(zero)

        max_len = self.args.max_length
        print('Loading TreeDataset {}'.format(len(loader)))
        for i, tree in enumerate(loader):

            #print('Loading Tree Dataset {}\r'.format(i), end='')
            rootindex = tree.rootindex
            edges = self.correct_edge(tree.edge_index, rootindex)
            node_batch = tree.batch
            src_sents = tree.x
            label = tree.y
            id_ = tree['id']

            # May discard some nodes to match max input length
            segs = self.find_seg_emb(src_sents)[:max_len]
            src = self.append_and_pad(src_sents, 'src')[:max_len]
            mask_src = src!=self.pad_vid

            # Only preserve the edges that in the valid sents
            num_valid_sent = (src==self.cls_vid).sum()
            edges = edges[:,edges[1]<num_valid_sent]
            node_batch = node_batch[:num_valid_sent]

            self.class_dict[label[0].item()] += 1

            list_ = [src, segs, mask_src, label, edges, node_batch, id_]
            examples.append(MyExample.fromlist(list_, fields))
        print('Finish Loading Tree Dataset')
        super(TreeDataset, self).__init__(examples, fields)
        #super(TreeDataset, self).__init__(args, loader, dataname, device, tokenizer, tgt_str)


    def sample_per_cls(self, num_label):
        output = []
        for n in range(num_label):
            output.append(self.class_dict[n])
        return output


    def append_and_pad(self, idx, mode):
        if mode == 'src':
            cnt_len = 0
            sents = []
            for x in idx:
                x = x[x!=self.pad_vid]
                x = x[:38]
                sent = torch.cat((self.cls_vid, x, self.sep_vid))
                cnt_len += len(sent)
                sents.append(sent)

            # Padding and truncating
            pad_len = max(self.args.max_length-cnt_len,0)
            output = F.pad(torch.cat(sents), (0, pad_len), value=self.pad_vid.item())
            return output

        elif mode == 'tgt':
            sent = torch.cat((self.tgt_bos, idx[:38], self.tgt_eos))
            # padding and truncating
            pad_length = 40-len(sent)
            output = F.pad(sent, (0,pad_length), value=self.pad_vid.item())
            output = output[:self.args.max_length_gen]
            return output

    def find_seg_emb(self, src_sents):
        flag = 0
        cnt_len = 0
        embs = []
        for sent in src_sents:
            sent = sent[sent!=self.pad_vid]
            if flag%2==0:
                pos_emb = torch.zeros(len(sent)+2).long()
            else:
                pos_emb = torch.zeros(len(sent)+2).long()+1
            cnt_len += (len(sent)+2)
            flag += 1
            embs.append(pos_emb)
        # padding and trucating 
        flag -= 1
        if flag%2==0:
            pos_emb = torch.zeros(max(self.args.max_length-cnt_len,0)).long()
        else:
            pos_emb = torch.zeros(max(self.args.max_length-cnt_len,0)).long()+1
        embs.append(pos_emb)
        segs = torch.cat(embs)[:self.args.max_length]
        return segs

    def correct_edge(self, edges, rootindex):
        parents  = edges[0]
        children = edges[1]
        parents[parents>children] = rootindex
        return torch.stack((parents, children), dim=0)
    


class Iterator(Iterator):

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield MyBatch(minibatch, self.dataset, self.device)
            if not self.repeat:
                return


class MyBatch(Batch):
    def __init__(self, data=None, dataset=None, device=None):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names
            self.input_fields = [k for k, v in dataset.fields.items() if
                                 v is not None]


            for (name, field) in dataset.fields.items():
                if field == 'string':
                    batch = [getattr(x, name) for x in data]
                    setattr(self, name, batch)
                                        
                elif field is not None:
                    batch = [getattr(x, name) for x in data]
                    #setattr(self, name, batch)
                    setattr(self, name, field.process(batch, device=device))
    def __iter__(self):
        yield self._get_field_values(self.input_fields)
        #yield self._get_field_values(self.target_fields)

class MyExample(Example):
    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field == 'string':
                if isinstance(val, str):
                    setattr(ex, name, val)
            elif field == 'Done':
                setattr(ex, name, val)
            elif field is not None:
                if isinstance(val, str):
                    val = val.rstrip('\n')
                # Handle field tuples
                if isinstance(name, tuple):
                    for n, f in zip(name, field):
                        setattr(ex, n, f.preprocess(val))
                else:
                    setattr(ex, name, field.preprocess(val))
        return ex

class MyField(Field):
    def process(self, batch, device=None):
        output = torch.stack(batch).to(device)
        return output

class PassField(Field):
    def process(self, batch, device=None):
        return batch
'''
    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explicitly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, str)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)
        try:
            var = torch.tensor(arr, dtype=self.dtype, device=device)
        except:
            pdb.set_trace()

        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var
'''
