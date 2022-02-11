import random
import os, sys
sys.path.append(os.getcwd())
import argparse
import pdb
#import wandb
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

from for_demo.build_new_data import build_from_new_input

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
    parser.add_argument('-for_demo', action='store_true')
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
    parser.add_argument("-inference_file", default=None, type=str)
    # Architecture setting
    parser.add_argument("-fold", default='', type=str,)
    #parser.add_argument("-fold", default=0, type=int, help='specify the data fold')
    #parser.add_argument("-num_detector", default=1, type=int)
    #parser.add_argument("-add_sibling", type=str2bool, nargs='?', const=True,default=False)
    #parser.add_argument("-skip_connect", type=str2bool, nargs='?', const=True,default=True)
    #parser.add_argument("-cond_type", default='', type=str)
    #parser.add_argument('-cond_loss', type=str2bool, nargs='?',const=True, default=False)
    #parser.add_argument("-variance", default=0.0001, type=float)
    #parser.add_argument("-no_duplicate", type=str2bool, nargs='?', const=True,default=False)
    #parser.add_argument("-hard_cond", type=str2bool, nargs='?', const=True,default=False)
    parser.add_argument('-model_file', default='all', type=str, help='test which check point ot test all')
    ## For Transformer
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    #parser.add_argument("-result_path", default='../save_result/Pheme/debug/result')
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

    #parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    #parser.add_argument("-test_from", default='')
    #parser.add_argument("-test_start_from", default=-1, type=int)

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
    if args.debug or args.run_exp or args.test_detector or args.early:
        wandb = None
    else:
        #wandb.init(project='RumorGD_{}_{}'.format(args.dataset_name, args.fold), 
        #           name=args.savepath.split('/')[-1])
        #config = wandb.config.update(args)
        wandb = None

    # Give random seed for reproduce
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device('cuda:{}'.format(args.visible_gpus))
    torch.cuda.manual_seed(args.seed)

    # The running experiments   
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

            if args.inference_file is None:
                x_test, x_train = loadfoldlist(args.dataset_name, fold, quat)
                x_val = x_test
                text_df, label_df = None, None
            else:
                # Whether to use the on-time inference file
                logger.info("We inference the model with new input line")
                text_df, label_df = build_from_new_input(args.inference_file, args.dataset_dir)
                x_test = label_df[1].tolist()
                x_test = [str(i) for i in x_test] 
                x_train = x_test
                x_val = x_test

            ###
            ### Train Detector
            ###
            # Don't drop the edges when training generator

            # Build model and train model
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            trainer = RumorTrainer(args, args.savepath, tokenizer, iter, fold, wandb)

            if args.for_demo:
                treeDic = buildgraph(args.dataset_name, 'txt_emb', args.label_num, texts=text_df, labels=label_df)
                test_list = loadBiTextData(args.dataset_name+'text', treeDic, x_test, tokenizer, data_path=None)
                test_loader = DataLoader(test_list, batch_size=1, shuffle=False, num_workers=5)
                trainer.inference_for_demo(test_loader, run_gen=args.test_gen or args.test_adv, level=1 if args.inference_file else 3)


            elif args.early:
                """ For early test, only change the graph data (different time interval)""" 
                portions = args.early.split(',')
                for port in portions:

                    # Build textgraph data for early detection
                    treeDic = buildgraph(args.dataset_name, 'txt_emb', args.label_num, port+'.' )

                    # Load data.TD_RvNN.vol_5000.txt
                    #treeDic = loadTree(args.dataset_name, port)

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
                treeDic = buildgraph(args.dataset_name, 'txt_emb', args.label_num, texts=text_df, labels=label_df)

                # Load data.TD_RvNN.vol_5000.txt
                #treeDic = loadTree(args.dataset_name)

                # Load textgraph
                train_list = loadBiTextData(args.dataset_name+'text', treeDic, x_train, tokenizer, 0.2, 0.2)
                val_list = loadBiTextData(args.dataset_name+'text', treeDic, x_val, tokenizer, 0.2, 0.2)
                test_list = loadBiTextData(args.dataset_name+'text', treeDic, x_test, tokenizer, data_path=data_path)

                train_loader = DataLoader(train_list, batch_size=1, shuffle=False, num_workers=5)
                val_loader = DataLoader(val_list, batch_size=1, shuffle=False, num_workers=5)
                test_loader = DataLoader(test_list, batch_size=1, shuffle=False, num_workers=5)

            # Record train, val, test list
            #logger.info('Tree number: train:{}, val:{}, test:{}'.format(
            #    len(train_list), len(val_list), len(test_list)))

            if args.run_exp:
                trainer.exp(test_loader, exp_pos=True)

            elif args.train_detector:
                trainer.forward(train_loader, val_loader, test_loader)

            elif args.test_detector:
                trainer.test_detector(test_loader, run_gen=args.test_gen or args.test_adv)

            if args.train_adv:
                args.train_gen = True
                trainer.train_adv(train_loader, val_loader, test_loader)

            elif args.train_gen:
                trainer.forward(train_loader, val_loader, test_loader)

            elif args.build_data:
                trainer.build_data(test_loader)


    '''

    if True:
        fold = args.fold
        print('Train fold ', args.fold)
        x_test, x_train = loadfoldlist(args.dataset_name, fold)
        if args.det_use_text:
            """
            Using word embedding as the input
            """
            # Method2: Build from pretrined BERT ----------------
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
            train_list, test_list = loadBiTextData(args.dataset_name+'text', treeDic, x_train, x_test, tokenizer,\
                args.det_td_droprate, args.det_bu_droprate)
            vocab_vectors = BertModel.from_pretrained('bert-base-uncased', cache_dir=args.temp_dir).embeddings.word_embeddings.weight
        else:
            """
            Using idx:cnt as the input
            """
            train_list, test_list = loadBiData(args.dataset_name, treeDic, x_train, x_test, args.det_td_droprate, args.det_bu_droprate)
            vocab_vectors = None
            tokenizer = None


        #################################################
        ## Train the Detectors (and Create Conditions) ##
        #################################################

        if args.train_detector:
            #with open(args.savepath+'/average_performance.tsv', 'a') as f:
            #    f.write('State---\tI\tF\tAcc\tAcc1\tPre1\tRec1\tF1\tAcc2\tPre2\tRec2\tF2\tEp\tCostTime\n')
            paths = []
            detector_path = args.detector_path
            for i in range(args.num_detector):
                print('Training Rumor Detector')
                args.detector_path = detector_path+'/{}'.format(i)
                train_loader = DataLoader(train_list, batch_size=args.det_bs, shuffle=True, num_workers=5)
                test_loader =  DataLoader(test_list, batch_size=args.det_bs, shuffle=False, num_workers=5)
                os.makedirs(args.detector_path, exist_ok=True)

                if args.make_cond: # we do not want the root enhance when creating the condition
                    det_trainer = RumordetTrainer(args, args.detector_path, vocab_vectors, iter, fold, wandb, root_enhance=False)
                else:
                    det_trainer = RumordetTrainer(args, args.detector_path, vocab_vectors, iter, fold, wandb, root_enhance=True)
                det_trainer.train(train_loader, test_loader)
                if args.make_cond:
                    from analysis.anal_cond import find_percentage
                    # Assign the drop rate to 0 to avoid dropping any edge (When making condition, every edge is needed)
                    train_list, test_list = loadBiTextData(args.dataset_name+'text', treeDic, x_train, x_test, tokenizer, 0, 0)
                    train_loader = DataLoader(train_list, batch_size=args.det_bs, shuffle=False, num_workers=5)
                    test_loader =  DataLoader(test_list, batch_size=args.det_bs, shuffle=False, num_workers=5)

                    comGenerator = CommentGenerator(args, args.savepath, tokenizer, vocab_vectors, iter, fold, wandb)
                    prefix_path = '/'.join(cache_path.split('/')[:-1])
                    
                    prefix_path_idx = prefix_path+'/{}/cond'.format(i)
                    comGenerator.obtain_cond(train_loader, prefix_path_idx)
                    comGenerator.obtain_cond(test_loader,  prefix_path_idx)
    
                    percent = find_percentage(prefix_path_idx)
                    with open(args.detector_path+'/average_performance.tsv', 'a') as f:
                        f.write('[All] % of comments in the same direction: {:.2f}%\n'.format(percent))
                    paths.append(prefix_path_idx)

            if args.make_cond:
                from Process.format_cond import merge_multi_conditions
                merge_multi_conditions(paths, cache_path, mode='vote') # Merge condition by voting
            args.make_cond = False

        if False:
            # For toy example
            from Process.toy_cond import create_toy_condition
            data_path = os.path.join('./data', args.dataset_name + 'textgraph')
            x_list = x_train + x_test
            create_toy_condition(x_list, data_path, args.cache_path)

        ###########################
        ## Create the conditions ##
        ###########################

        if args.make_cond:
            from analysis.anal_cond import find_percentage
            # Assign the drop rate to 0 to avoid dropping any edge
            train_list, test_list = loadBiTextData(args.dataset_name+'text', treeDic, x_train, x_test, tokenizer, 0, 0)
            # Load data for comment generator
            train_loader = DataLoader(train_list, batch_size=args.det_bs, shuffle=False, num_workers=5)
            test_loader =  DataLoader(test_list, batch_size=args.det_bs, shuffle=False, num_workers=5)
            comGenerator = CommentGenerator(args, args.savepath, tokenizer, vocab_vectors, iter, fold, wandb) 
            comGenerator.obtain_cond(train_loader, cache_path)
            comGenerator.obtain_cond(test_loader,  cache_path)

            percent = find_percentage(cache_path)

            with open(args.detector_path+'/average_performance.tsv', 'a') as f:
                f.write('[All] % of comments in the same direction: {:.2f}%\n'.format(percent))
            args.make_cond = False

        ########################
        ## Train the Attacker ##
        ########################

        if args.train_attacker:
            os.makedirs(args.savepath, exist_ok=True)
            train_list, test_list = loadBiTextData(args.dataset_name+'text', treeDic, x_train, x_test, tokenizer, 0, 0)
            # Load data for comment generator (Require batch size to be one)
            train_loader = DataLoader(train_list, batch_size=1, shuffle=False, num_workers=5)
            test_loader =  DataLoader(test_list, batch_size=1, shuffle=False, num_workers=5)
            # Train Comment Generator
            comGenerator = CommentGenerator(args, args.savepath, tokenizer, vocab_vectors, iter, fold, wandb) 
            comGenerator.begin(train_loader, test_loader)


        #####################
        ## Calculate ROUGE ##
        #####################

        if args.testing:
            from tools.eval_cond import eval_cond_result
            args.result_path = os.path.join(args.savepath, 'result')
            test_loader =  DataLoader(test_list, batch_size=1, shuffle=False, num_workers=5)
            comGenerator = CommentGenerator(args, args.savepath, tokenizer, vocab_vectors, iter, fold, wandb)
            comGenerator.only_test(test_loader, cache_path)

            eval_cond_result(args.result_path, 'gold')
            eval_cond_result(args.result_path, 'candidate')

        ##############################################
        ## Augment Dataset and Train Detector Again ##
        ##############################################

        if args.format_new_dataset:
            #from tools.eval_cond import eval_cond # only for toy examples
            train_list, test_list = loadBiTextData(args.dataset_name+'text', treeDic, x_train, x_test, tokenizer, 0, 0)
            train_loader = DataLoader(train_list, batch_size=1, shuffle=False, num_workers=5)
            test_loader =  DataLoader(test_list, batch_size=1, shuffle=False, num_workers=5)

            # Obtain the prediction result and save to the '/temp' by id -> [parent_idx, self_idx, comment]
            comGenerator = CommentGenerator(args, args.savepath, tokenizer, vocab_vectors, iter, fold) 
            if args.model_file=='all':
                models = []
                for i in range(args.train_epoch):
                    model_file = '{}_model.pt'.format(i)
                    if os.path.isfile(os.path.join(args.savepath,model_file)):
                        models.append(model_file)
                model_file = 'best_model.pt'
                if os.path.isfile(os.path.join(args.savepath, model_file)):
                    models.append(model_file)
            else:
                models = [args.model_file]

            for model_file in models:
                comGenerator.inference_save_by_id(test_loader, cache_dir=cache_path, model_file=model_file)
                #comGenerator.merge_nodes(test_loader, tokenizer, cache_dir=cache_path)
            
                #comGenerator.inference_save_by_id(train_loader, cache_dir=cache_path, model_file=args.model_file)
                #comGenerator.merge_nodes(train_loader, tokenizer, cache_dir=cache_path)
                del comGenerator

                """ # It is for toy example
                cnt_success, cnt = eval_cond(args.savepath+'/temp')
                with open(args.savepath+'/cond.txt', 'a') as f:
                    f.write('{}\n'.format(model_file))
                    f.write('{}/{}({:.2f}%)\n'.format(cnt_success, cnt, 100*cnt_success/cnt))
                """
                """
                with open(args.savepath+'/average_performance.tsv', 'a') as f:
                    f.write('--Train by augmented dataset (from {} with prefer {}) and test by the original dataset--\n'.format(args.model_file, args.prefer))
                for i in range(10):
                    if args.det_use_text:
                        # The train data is from the augmented 
                        train_list, _ = loadBiTextData(args.dataset_name+'text', treeDic, x_train, x_test, tokenizer, args.det_td_droprate, args.det_bu_droprate, data_path=args.savepath+'/'+args.dataset_name+'textgraph')
                        # The test data is from the original
                        _, test_list = loadBiTextData(args.dataset_name+'text', treeDic, x_train, x_test, tokenizer, args.det_td_droprate, args.det_bu_droprate)

                    else:
                        train_list, _ = loadBiData(args.dataset_name, treeDic, x_train, x_test, args.det_td_droprate, args.det_bu_droprate, data_path=args.savepath+'/'+args.dataset_name+'graph')
                        test_list, _ = loadBiData(args.dataset_name, treeDic, x_train, x_test, args.det_td_droprate, args.det_bu_droprate)
                        vocab_vectors = None
                        tokenizer = None


                    train_loader = DataLoader(train_list, batch_size=int(args.det_bs/2), shuffle=False, num_workers=5)
                    test_loader =  DataLoader(test_list, batch_size=int(args.det_bs/2), shuffle=False, num_workers=5)

                    # Train Rumor Detector again
                    print('Train Rumor Detector for 10 timse, processing: {}/10'.format(i))
                    det_trainer = RumordetTrainer(args, args.savepath, vocab_vectors, i, fold, wandb, root_enhance=True)
                    det_trainer.train(train_loader, test_loader)
                """
        
        #############################
        ## Plot t-SNE for features ##
        #############################

        if args.plot_feat:
            test_loader =  DataLoader(test_list, batch_size=1, shuffle=False, num_workers=5)
            comGenerator = CommentGenerator(args, args.savepath, tokenizer, vocab_vectors, iter, fold, wandb)
            comGenerator.plot_feat(test_loader, cache_path, args.model_file)

    #cal_average_performance(args.savepath+'/average_performance.tsv')
    print(args)
    '''
