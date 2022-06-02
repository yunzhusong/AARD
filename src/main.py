import os, sys
sys.path.append(os.getcwd())
import argparse
import pdb
import pprint
from torch_geometric.data import DataLoader
from shutil import copyfile

from data.rand5fold import *
from data.process import *
from data.getgraph import buildgraph

from models.trainer_gen import RumorTrainer

from transformers import BertModel, BertTokenizer
import numpy as np
import torch

import shutil
import time

from others.logging import init_logger, logger

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def logged_args(args):
    args_dict = copy.deepcopy(vars(args))
    del args_dict['debug']
    del args_dict['encoder']
    del args_dict['temp_dir']
    del args_dict['visible_gpus']
    del args_dict['gpu_ranks']
    del args_dict['log_file']

def parse_args():
    parser = argparse.ArgumentParser(description='For Pheme dataset')
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-savepath', type=str, help='where to save the attacker', default='../results/debug')
    parser.add_argument('-textgraph_dir', type=str, help='whether to specific text graph', default=None)
    parser.add_argument('-cache_path', type=str, help='where to save the condition files')
    ## Switch what to do
    parser.add_argument("-plot_feat", type=str2bool, nargs='?', const=True,default=False, help='plot t-sne for features')
    parser.add_argument("-run_exp", action='store_true')
    parser.add_argument("-train_detector", action='store_true')
    parser.add_argument("-test_detector", action='store_true')
    parser.add_argument("-train_gen", action='store_true')
    parser.add_argument("-test_gen", action='store_true')
    parser.add_argument("-test_adv", action='store_true')
    parser.add_argument("-train_adv", action='store_true')
    parser.add_argument("-build_data", action='store_true')
    parser.add_argument("-early", type=str, default='')
    parser.add_argument("-quat", type=str, default='')
    parser.add_argument("-filter", type=str2bool, default=True)
    parser.add_argument("-label_num", type=int, default=2)
    parser.add_argument("-train_gen_from", default='../results/pretrain/XSUM_BertExtAbs/')

    # Architecture setting
    parser.add_argument("-fold", default='', type=str, help='specify the data fold')
    parser.add_argument('-model_file', default='all', type=str, help='test which check point ot test all')

    ## For Transformer
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-temp_dir", default='./temp')

    parser.add_argument("-batch_size", default=4, type=int)
    parser.add_argument("-train_epoch", default=40, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=True) 
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)

    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=True) #
    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=2, type=int)
    parser.add_argument("-min_length", default=1, type=int)
    parser.add_argument("-max_length", default=300, type=int)
    parser.add_argument("-max_length_gen", default=40, type=int)
    parser.add_argument("-min_length_gen", default=1, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=2000, type=int)
    parser.add_argument("-warmup_steps_bert", default=2000, type=int)
    parser.add_argument("-warmup_steps_dec", default=2000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_epoch", default=30, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=2, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('-visible_gpus', default='0', type=str, help='train on which gpu')
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='./save_model/Pheme/debug/result.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-dataset_name", type=str, default='')
    parser.add_argument("-dataset_dir", type=str, default='')
    parser.add_argument("-det_input_dim", type=int, default=5000)
    parser.add_argument("-det_hid_dim", type=int, default=64)
    parser.add_argument("-det_out_dim", type=int, default=64)
    parser.add_argument("-det_max_epoch", type=int, default=200)
    parser.add_argument("-det_lr", type=float, default=0.0005)
    parser.add_argument("-det_weight_decay", type=float, default=1e-4)
    parser.add_argument("-det_patience", type=int, default=7)
    parser.add_argument("-det_bs", type=int, default=128)
    parser.add_argument("-det_td_droprate", type=float, default=0.2)
    parser.add_argument("-det_bu_droprate", type=float, default=0.2)
    parser.add_argument("-det_use_text", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-det_train_text", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-det_text_emb_dim", type=int, default=768)

    parser.add_argument("-log_tensorboard", action='store_true')
    

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)

    return args


if __name__=='__main__':
    args = parse_args()

    if args.dataset_name == '':
        args.dataset_name = args.dataset_dir.split('/')[-1]

    cache_path = args.cache_path

    data_dir = args.dataset_dir

    # Give random seed for reproduce
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device('cuda:{}'.format(args.visible_gpus))
    torch.cuda.manual_seed(args.seed)

    iter = 0 
    folds = args.fold.split(',')
    quats = args.quat.split(',')
    savepath = args.savepath

    for fold in folds:
        for quat in quats:
            print('Train Fold [{}] Quantity [{}]'.format(fold, quat))

            # build arguments
            args.fold = fold
            args.savepath = savepath+'/{}/{}'.format(fold, quat)
            args.log_file = os.path.join(args.savepath, 'result.log')
            os.makedirs(args.savepath, exist_ok=True)

            # Initial logger
            init_logger(args.log_file)
            logger.info(str(args))

            if quat == '100':
                quat = ''

            x_test, x_train = loadfoldlist(args.dataset_name, fold, quat)
            num_train_tree = len(x_train)
            x_val = x_test
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            ######################
            ### Train Detector ###
            ######################
            # Not drop the edges when training generator

            # Build model and train model
            trainer = RumorTrainer(args, args.savepath, tokenizer, iter, fold)

            if args.early:
                """ For early test, only change the graph data (different time interval)""" 
                portions = args.early.split(',')
                for port in portions:

                    # Build textgraph data for early detection
                    treeDic = buildgraph(args.dataset_name, 'txt_emb', args.label_num, '{}.'.format(port))

                    # Load textgraph
                    test_list = loadBiTextData(args.dataset_name+'text',
                                               treeDic, x_test, tokenizer)
                    test_loader = DataLoader(test_list, batch_size=1, shuffle=False, num_workers=5)
                    trainer.test_detector(test_loader, run_gen=args.test_gen, portion=port)
            else:
                if args.textgraph_dir is not None:
                    data_path = args.textgraph_dir + '/{}/twitter16_pollute_textgraph'.format(fold)
                else:
                    data_path = None

                # Build textgraph data
                treeDic = buildgraph(args.dataset_name, 'txt_emb', args.label_num)

                # Load textgraph
                train_list = loadBiTextData(args.dataset_name+'text', treeDic, x_train, tokenizer, 0.2, 0.2)
                val_list = loadBiTextData(args.dataset_name+'text', treeDic, x_val, tokenizer, 0.2, 0.2)
                test_list = loadBiTextData(args.dataset_name+'text', treeDic, x_test, tokenizer, data_path=data_path)

                train_loader = DataLoader(train_list, batch_size=1, shuffle=False, num_workers=5)
                val_loader = DataLoader(val_list, batch_size=1, shuffle=False, num_workers=5)
                test_loader = DataLoader(test_list, batch_size=1, shuffle=False, num_workers=5)

            if args.run_exp:
                trainer.exp(test_loader, exp_pos=True)

            elif args.train_detector:
                trainer.forward(train_loader, val_loader, test_loader)

            elif args.test_detector:
                trainer.test_detector(test_loader, run_gen=args.test_gen)

            if args.train_adv:
                args.train_gen = True
                trainer.train_adv(train_loader, val_loader, test_loader)

            elif args.train_gen:
                trainer.forward(train_loader, val_loader, test_loader)

            elif args.build_data:
                trainer.build_data(test_loader)


