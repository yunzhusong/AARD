import sys, os
sys.path.append(os.getcwd())
sys.path.insert(0, './pretrain')
import pdb
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from torch.distributions.normal import Normal
from data.process import *
from data.rand5fold import *
from others.optimizers import Optimizer

#from modelsmy.transformers import BertModel
from models.model_decoder import TransformerDecoder
from models.model_detector import GCNClassifier

from transformers import BertConfig
from models.mybert import MyBertModel

def build_optim(args, model, checkpoint=None):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)
    optim.set_parameters(list(model.named_parameters()))

    return optim


def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)
    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)

    if checkpoint is not None:
        saved_optimizer_state_dict = checkpoint['optims'][0]
        #saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to('cuda:{}'.format(args.visible_gpus))

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """
    optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)

    if checkpoint is not None:
        saved_optimizer_state_dict = checkpoint['optims'][1]
        #saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to('cuda:{}'.format(args.visible_gpus))

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func)
    generator.to(device)

    return generator


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = MyBertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = MyBertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x=None, segs=None, mask=None, embedding_output=None, 
                return_dict=True, return_embout=False):

        if(self.finetune):
            top_vec = self.model(x, token_type_ids=segs, attention_mask=mask,
                                 embedding_output=embedding_output,
                                 return_dict=return_dict, 
                                 return_embedding_output=return_embout)
        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(x, token_type_ids=segs, attention_mask=mask,
                                     embedding_output=embedding_output,
                                     return_dict=return_dict, 
                                     return_embedding_output=return_embout)
        return top_vec
        #if not return_embout:
        #    return top_vec[0]
        #else:
        #    return (top_vec[0], top_vec[2])


class RumorDetector(nn.Module):
    def __init__(self, args, device, symbols=None, checkpoint=None, checkpoint_gen=None):
        super(RumorDetector, self).__init__()
        self.args = args
        self.device = device
        self.symbols = symbols
        # Build encoder
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        # Build Classifier
        hidden_dim = self.bert.model.config.hidden_size
        self.classifier = GCNClassifier(0.1, hidden_dim, args.label_num, args.filter)

        ###
        ### Use Generation Branch
        ###
        if args.train_gen or args.test_gen or args.test_adv or args.build_data or args.train_adv:
            # Initial the decoder's final layer
            vocab_size = self.bert.model.config.vocab_size
            shared_emb = self.bert.model.embeddings.word_embeddings.weight
            self.vocab_size = vocab_size
            tgt_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

            if (self.args.share_emb):
                # Share encoder's word embedding to decoder's final layer
                tgt_embeddings.weight = copy.deepcopy(shared_emb)

            # Build decoder
            self.decoder = TransformerDecoder(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.dec_heads,
                d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout,
                embeddings=tgt_embeddings)

            self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
            self.generator[0].weight = self.decoder.embeddings.weight

            # Load checkpoint or initialization the decoder and generator
            if checkpoint_gen is not None:
                """ including weights of bert and decoder """
                self.load_state_dict(checkpoint_gen['model'], strict=False)
            else:
                for module in self.decoder.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        module.weight.data.normal_(mean=0.0, std=0.02)
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()
                for p in self.generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                    else:
                        p.data.zero_()
                if(args.use_bert_emb):
                    tgt_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
                    tgt_embeddings.weight = copy.deepcopy(shared_emb)
                    self.decoder.embeddings = tgt_embeddings
                    self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, segs, mask_src, edges=None, node_batch=None, 
                tgt=None, mask_tgt=None, gen_flag=True, gen_testing=False):

        """
        Args:
            tgt (`torch.Tensor` of shape :obj:(batch_size, seq_length))
        """
        gen = (self.args.train_gen or self.args.test_gen or self.args.test_adv) and gen_flag
        if gen and tgt is not None:
            """ for training generator and detector"""
            # encode
            with torch.no_grad():
                top_vec = self.bert(src, segs, mask_src, return_embout=True)
            last_hidden_state = top_vec.last_hidden_state
            emb_input = top_vec.embedding_output
            sent_pos = top_vec.sent_pos

            # decode
            dec_state = self.decoder.init_decoder_state(src, last_hidden_state)
            decoder_outputs, state, weight = self.decoder(tgt[:,:-1], last_hidden_state,
                                     dec_state, emb_input, sent_pos, testing=gen_testing)
            # classify
            emb_output = torch.cat((emb_input, decoder_outputs), dim=1)

            # detector
            top_vec = self.bert(embedding_output=emb_output)
            top_vec.sent_pos = sent_pos

            detector_logits = self.classifier(top_vec, edges, node_batch, weight).logits
            return (detector_logits, decoder_outputs)

        elif gen and tgt is None:
            """ for testing generator and detector without comment target"""
            # prepare for decoder predicting
            max_length = self.args.max_length_gen
            min_length = self.args.min_length_gen
            start_token = self.symbols['BOS']
            end_token = self.symbols['EOS']
            pad_embedding = self.bert.model.embeddings.word_embeddings.weight[start_token]

            # encode
            with torch.no_grad():
                top_vec = self.bert(src, segs, mask_src, return_embout=True)
            src_feature = top_vec.last_hidden_state
            src_embedding = top_vec.embedding_output
            sent_pos = top_vec.sent_pos

            # decode
            dec_state = self.decoder.init_decoder_state(src, src_feature, with_cache=True)
            decoder_outputs, _, weight = self.decoder.predict(src_feature, src_embedding, 
                sent_pos, dec_state, self.generator, max_length, min_length, 
                start_token, end_token, pad_embedding)

            # classify
            output_embedding = torch.cat((src_embedding, decoder_outputs), dim=1)

            # detector
            top_vec = self.bert(embedding_output=output_embedding)
            top_vec.sent_pos = sent_pos

            detector_logits = self.classifier(top_vec, edges, node_batch, weight).logits
            return (detector_logits, decoder_outputs)


        else:
            """ for only training detector """
            top_vec = self.bert(src, segs, mask_src)
            detector_logits = self.classifier(top_vec, edges, node_batch).logits
            return (detector_logits,)

    def exp_pos(self, src, segs, mask_src, weight, edges=None, node_batch=None, 
                tgt=None, mask_tgt=None):
        """ for runing experiment of position importance
        weight: control manually
        """
        # batch size should be 1
        assert len(src) == 1

        # prepare for decoder predicting
        max_length = self.args.max_length_gen
        min_length = self.args.min_length_gen
        start_token = self.symbols['BOS']
        end_token = self.symbols['EOS']
        pad_embedding = self.bert.model.embeddings.word_embeddings.weight[start_token]

        # encode
        with torch.no_grad():
            top_vec = self.bert(src, segs, mask_src, return_embout=True)
        src_feature = top_vec.last_hidden_state
        src_embedding = top_vec.embedding_output
        sent_pos = top_vec.sent_pos

        # decode
        dec_state = self.decoder.init_decoder_state(src, src_feature, with_cache=True)
        decoder_outputs, _, _ = self.decoder.predict(src_feature, src_embedding,
            sent_pos, dec_state, self.generator, max_length, min_length, 
            start_token, end_token, pad_embedding)

        # classify
        output_embedding = torch.cat((src_embedding, decoder_outputs), dim=1)

        # detector
        top_vec = self.bert(embedding_output=output_embedding)
        top_vec.sent_pos = sent_pos

        detector_logits = self.classifier(top_vec, edges, node_batch, weight).logits
        return (detector_logits, decoder_outputs)


