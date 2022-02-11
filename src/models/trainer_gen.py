import os, sys
sys.path.append(os.getcwd())
import argparse
import time
import json
import shutil
import torch
import torch.nn as snn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from eval.evaluate import *
from data.loader import CommentDataset, TreeDataset, Iterator
from torchtext.legacy.data import BucketIterator

from models.model import RumorDetector, build_optim_bert, build_optim_dec, build_optim
from models.trainer import build_trainer
from models.predictor import build_predictor
from others.earlystopping2class import EarlyStopping
from others.loss import abs_loss
from others.logging import logger

from tqdm import tqdm
from joblib import Parallel, delayed
import random

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

class RumorTrainer(object):
    def __init__(self, args, savepath, tokenizer=None, iter=0, fold=0, wandb=None):
        super(RumorTrainer)
        self.args = args
        self.fold = args.fold
        self.savepath = savepath
        self.cache_path = args.cache_path
        self.tokenizer = tokenizer
        self.iter = iter
        self.fold = fold
        self.dataname = args.dataset_name
        self.device = 'cuda:{}'.format(self.args.visible_gpus)
        self.maxepoch = args.train_epoch
        self.bs = args.batch_size
        self.wandb = wandb

        #if not self.args.run_exp:
        #    # Log information into a txt file
        #    init_logger(args.log_file)
        #    logger.info(str(args))
        # Specify random seed for reproduce
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

        ## Build trainer -----------------------
        # 1. Load a checkpoint for generation brach 
        if args.train_gen and args.train_gen_from != '':
            logger.info('Loading generation branch checkpoint from %s' % args.train_gen_from)

            pretrained = torch.load(args.train_gen_from+'step_30000.pt', map_location=lambda storage, loc: storage)
            checkpoint_gen = {'model': pretrained}
        else:
            checkpoint_gen = None


        # 3. Build model
        self.symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],\
                       'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
        self.model = RumorDetector(args, self.device, self.symbols, checkpoint_gen=checkpoint_gen)

        # 4. Set up optimizer, can optimize encoder(bert) and decoder seperately
        if (args.sep_optim):
            checkpoint=None
            optim_bert = build_optim_bert(args, self.model, checkpoint)
            optim_dec  = build_optim_dec(args, self.model, checkpoint)
            self.optim = [optim_bert, optim_dec]
        else:
            self.optim = [build_optim(args, self.model, checkpoint)]

    def forward(self, train_loader, val_loader, test_loader, maxepoch=None):
        """ Normal training process """

        # Build data iterator for generator (One node on the tree is converted to one data)
        #train = TreeDataset(self.args, train_loader, self.dataname, self.device, self.tokenizer)
        train = CommentDataset(self.args, train_loader, self.dataname, self.device, self.tokenizer)
        
        sample_per_cls = None
        #sample_per_cls = train.sample_per_cls(self.args.label_num)

        train_iter = BucketIterator(train, sort_key=lambda x: len(x.src),
            sort_within_batch=False, batch_size=self.bs, device=self.device)

        #val = TreeDataset(self.args, val_loader, self.dataname, self.device, self.tokenizer)
        val = CommentDataset(self.args, val_loader, self.dataname, self.device, self.tokenizer)
        val_iter = BucketIterator(val, sort_key=lambda x: len(x.src),
            sort_within_batch=False, batch_size=96, device=self.device)

        test = TreeDataset(self.args, test_loader, self.dataname, self.device, self.tokenizer)
        test_iter = Iterator(test, train=False, device=self.device, batch_size=96,
            sort_key=lambda x: len(x.src), sort_within_batch=False)

        # Define trainer
        if self.args.train_gen:
            train_loss = abs_loss(self.args.label_num, self.maxepoch, self.device, True,
                self.args.train_gen ,self.model.generator, self.symbols,
                self.model.vocab_size, self.args.label_smoothing, sample_per_cls=sample_per_cls)
        else:
            train_loss = abs_loss(self.args.label_num, self.maxepoch, self.device, True, sample_per_cls = sample_per_cls)

        trainer = build_trainer(self.args, self.model, self.optim, train_loss, wandb=self.wandb)

        # Start training
        best_loss = 10
        stop_count = 0
        tot_train_steps = self.maxepoch * len(train_iter)
        gen_flag = self.args.train_gen or self.args.test_gen
        if not maxepoch:
            maxepoch = self.maxepoch
        logger.info('Start training')
        for epoch in range(maxepoch):
            print('[Training] {}/{}'.format(epoch, self.maxepoch))

            if self.args.train_gen:
                job = 'gen det'
            else:
                job = 'det'
            message = '{}-{} epoch {} {}/{} '.format(self.dataname, self.fold, job, epoch, self.maxepoch)
            trainer.train(train_iter, tot_train_steps, message)
            val_stats = trainer.validate(val_iter, epoch)
            test_stats = trainer.testing(test_iter, epoch, gen_flag=gen_flag)
            test_stats.write_results(os.path.join(self.args.savepath, 'result_test.csv'), str(epoch), self.args.label_num)

            val_det_loss = val_stats.det_xent()
            # Save best model
            if val_det_loss < best_loss:
                print('Save model at epoch {}'.format(epoch))
                trainer._save('best')
                best_loss = val_det_loss
                stop_count = 0
            else:
                stop_count += 1

    def train_adv(self, train_loader, val_loader, test_loader):
        """ Adversarially training process"""
        ## Step1. Train detector and generator
        best_model = os.path.join(self.args.savepath, 'best_model.pt')
        if not os.path.exists(best_model):
            self.forward(train_loader, val_loader, test_loader, maxepoch=3)
        logger.info('Loading pre-trained genearator and detector')
        self.model.load_state_dict(torch.load(best_model, map_location=lambda storage, loc: storage)['model'])


        ## Step2. Train adv generator and fix detector
        # Load checkpoint
        best_adv_model = os.path.join(self.args.savepath, 'best_adv_model.pt')
        if os.path.exists(best_adv_model):
            logger.info('Loading pre-trained adv genearator')
            self.model.load_state_dict(torch.load(best_adv_model, map_location=lambda storage, loc: storage)['model'])
        else:
            logger.info('Adversarially train generator -----------------------')
            train = CommentDataset(self.args, train_loader, self.dataname, self.device, self.tokenizer, reverse_label=True)
            val = CommentDataset(self.args, val_loader, self.dataname, self.device, self.tokenizer, reverse_label=True)
            train_iter, val_iter = BucketIterator.splits((train, val), sort_key=lambda x: len(x.src),
                sort_within_batch=False, batch_size=self.bs, device=self.device) # 3906, 977
            test = TreeDataset(self.args, test_loader, self.dataname, self.device, self.tokenizer)
            test_iter = Iterator(test, train=False, device=self.device, batch_size=self.bs,
                sort_key=lambda x: len(x.src), sort_within_batch=False)

            # Define trainer
            train_loss = abs_loss(self.args.label_num, self.maxepoch, self.device, True,
                    self.args.train_gen ,self.model.generator, self.symbols,
                    self.model.vocab_size, self.args.label_smoothing)
            trainer = build_trainer(self.args, self.model, self.optim, train_loss, wandb=self.wandb)

            tot_train_steps = self.maxepoch * len(train_iter)
            test_stats = trainer.testing(test_iter, -1, gen_flag=True)
            test_stats.write_results(os.path.join(self.args.savepath, 'result_test.csv'), 'Before', self.args.label_num)

            # Freeze the detector
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.decoder.parameters():
                param.requires_grad = True

            best_total_loss = 100
            lowest_acc = 1
            stop_count = 0
            for epoch in range(self.maxepoch):
                print('[Adv train generator - fold {}] {}/{}'.format(self.fold, epoch, self.maxepoch))

                message = '{}-{} epoch {}/{} adv gen'.format(self.dataname, self.fold, epoch, 4)
                trainer.train(train_iter, tot_train_steps, message)
                val_stats = trainer.validate(val_iter, epoch)
                test_stats = trainer.testing(test_iter, epoch, gen_flag=False)
                test_stats = trainer.testing(test_iter, epoch, gen_flag=True) # polluted data
                test_stats.write_results(os.path.join(self.args.savepath, 'result_test.csv'), '{}-att'.format(epoch), self.args.label_num)
                #Save best model
                if test_stats.det_acc() < lowest_acc:
                    logger.info('Save Adv model at epoch {}'.format(epoch))
                    lowest_acc = test_stats.det_acc()
                    trainer._save('best_adv')
                    stop_count = 0
                else:
                    stop_count += 1

                if stop_count == 3:
                    break

        
        ## Step3. Train adv detector and fix generator
        logger.info('Adversarially train detector ---------------------------')
        train = CommentDataset(self.args, train_loader, self.dataname, self.device, self.tokenizer)
        val = CommentDataset(self.args, val_loader, self.dataname, self.device, self.tokenizer)
        train_iter, val_iter = BucketIterator.splits((train, val), sort_key=lambda x: len(x.src),
            sort_within_batch=False, batch_size=self.bs, device=self.device) # 3906, 977
        test = TreeDataset(self.args, test_loader, self.dataname, self.device, self.tokenizer)
        test_iter = Iterator(test, train=False, device=self.device, batch_size=self.bs,
            sort_key=lambda x: len(x.src), sort_within_batch=False)

        # Load checkpoint
        best_adv_model = os.path.join(self.args.savepath, 'best_adv_model.pt')
        self.model.load_state_dict(torch.load(best_adv_model, map_location=lambda storage, loc: storage)['model'])

        # Freeze the generator
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        for param in self.model.bert.parameters():
            param.requires_grad = True

        # Define trainer
        optim  = [build_optim(self.args, self.model, None)]
        train_loss = abs_loss(self.args.label_num, self.maxepoch, self.device, True,
                False ,self.model.generator, self.symbols,
                self.model.vocab_size, self.args.label_smoothing)

        trainer = build_trainer(self.args, self.model, optim, train_loss, wandb=self.wandb)
        test_stats = trainer.testing(test_iter, -1, gen_flag=False) # clean data
        test_stats = trainer.testing(test_iter, -1, gen_flag=True) # polluted data

        tot_train_steps = self.maxepoch * len(train_iter)
        best_xent = 10
        stop_count = 0
        for epoch in range(self.maxepoch):
            print('[Adv train detector] {}/{}'.format(epoch, self.maxepoch))
            # Freeze the detector
            message = '{}-{} epoch {}/{} adv det'.format(self.dataname, self.fold, epoch, 5)
            trainer.train(train_iter, tot_train_steps, message)
            val_stats = trainer.validate(val_iter, epoch)
            test_stats = trainer.testing(test_iter, epoch, gen_flag=False) # clean data
            test_stats = trainer.testing(test_iter, epoch, gen_flag=True) # polluted data
            test_stats.write_results(os.path.join(self.args.savepath, 'result_test.csv'), '{}-adv'.format(epoch), self.args.label_num)

            # Save best model
            if val_stats.det_xent() < best_xent:
                print('Save model at epoch {}'.format(epoch))
                trainer._save('best_final')
                best_xent = val_stats.det_xent()
                stop_count = 0
            else:
                stop_count += 1

            if stop_count == 3:
                break

    def test_detector(self, loader, run_gen=False, portion='all'):
        """ Testing detector  """
        test = TreeDataset(self.args, loader, self.dataname, self.device, self.tokenizer)
        #test = CommentDataset(self.args, loader, self.dataname, self.device, self.tokenizer)

        data_iter = Iterator(test, train=False, device=self.device, 
                             batch_size=len(test) if len(test)<self.bs else self.bs,
                             sort_key=lambda x: len(x.src),
                             sort_within_batch=False)
        # Define trainer
        train_loss = abs_loss(self.args.label_num, self.maxepoch, self.device, train=False)
        trainer = build_trainer(self.args, self.model, self.optim, train_loss)

        logger.info('Test on best model (stage-1)')
        best_model = os.path.join(self.args.savepath, 'best_model.pt')

        if os.path.exists(best_model):
            try:
                self.model.load_state_dict(torch.load(best_model, map_location=lambda storage, 
                                                  loc: storage)['model'])
            except:
                self.model.load_state_dict(torch.load(best_model, map_location=lambda storage, 
                                                  loc: storage)['model'], strict=False)
                logger.info('[Warning] The keys in state dict do not strictly match')

            test_stat = trainer.testing(data_iter, tokenizer=self.tokenizer, gen_flag=False, info="Without Generated Response >>", write_type="w")
            test_stat.write_results(os.path.join(self.args.savepath, 'result_test.csv'), 'test-'+portion, self.args.label_num)

        if self.args.test_adv:

            logger.info('Test on adversarially-trained model (stage-2)')
            best_model = os.path.join(self.args.savepath, 'best_adv_model.pt')
            if os.path.exists(best_model):
                try:
                    self.model.load_state_dict(torch.load(best_model, map_location=lambda storage, 
                                                      loc: storage)['model'])
                except:
                    self.model.load_state_dict(torch.load(best_model, map_location=lambda storage, 
                                                      loc: storage)['model'], strict=False)
                    logger.info('[Warning] The keys in state dict do not strictly match')

            test_stat = trainer.testing(data_iter, tokenizer=self.tokenizer, gen_flag=True, info="\nWith Generated Response from {} >>".format(best_model.split("/")[-1]), write_type=="a")
            predictor = build_predictor(self.args, self.model, self.tokenizer, self.symbols, logger)
            predictor.translate(data_iter, 'best', have_gold=False)


    def build_data(self, loader):
        """ Build polluted textgraph by adversarial trained generator"""
        print("Build Pollued data by adv model from {}".format(self.args.savepath))
        test = TreeDataset(self.args, loader, self.dataname, self.device, self.tokenizer)
        data_iter = Iterator(test, train=False, device=self.device, batch_size=len(test),
                             sort_key=lambda x: len(x.src),
                             sort_within_batch=False)

        best_adv_model = os.path.join(self.args.savepath, 'best_adv_model.pt')
        self.model.load_state_dict(torch.load(best_adv_model, map_location=lambda storage, 
                                              loc: storage)['model'])
        predictor = build_predictor(self.args, self.model, self.tokenizer, self.symbols, logger)
        predictor.build(data_iter)

    def exp(self, loader, exp_pos=False):
        """ Testing detector  """
        test = TreeDataset(self.args, loader, self.dataname, self.device, self.tokenizer)
        data_iter = Iterator(test, train=False, device=self.device, batch_size=self.bs,
                             sort_key=lambda x: len(x.src),
                             sort_within_batch=False)

        best_model = os.path.join(self.args.savepath, 'best_adv_model.pt')

        # Define trainer
        loss = abs_loss(self.args.label_num, self.maxepoch, self.device, train=False)
        trainer = build_trainer(self.args, self.model, self.optim, loss)
        test_stats = trainer.testing(data_iter, -1)

        if exp_pos:
            if os.path.exists(best_model):
                self.model.load_state_dict(torch.load(best_model, map_location=lambda storage, 
                                                  loc: storage)['model'], strict=True)
                test_stat = trainer.exp_pos(data_iter)

    def inference_save_by_id(self, loader, cache_dir, model_file):

        datalist = CommentDataset(self.args, loader, cache_dir, self.dataname, self.device, self.tokenizer)
        data_iter = Iterator(datalist, batch_size=len(datalist), device=self.device, shuffle=False) # 1651

        if model_file:
            print('Model is loaded from ', model_file)
            if logger:
                logger.info('Model is from {}'.format(model_file))     
            model_path = os.path.join(self.savepath, model_file)
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)['model'], strict=False)
        else:
            print('Not loading pretrained model...')
            if logger:
                logger.info('Not loading pretrained model ...')
        try:
            print('[Warning] Going to delete the original temp file at ', self.args.savepath)
            shutil.rmtree(self.args.savepath+'/temp')
        except:
            print('Creating new temp folder')
        os.makedirs(os.path.join(self.args.savepath, 'temp'), exist_ok=True)
        os.makedirs(os.path.join(self.args.savepath, 'temp_gold'), exist_ok=True)
        self.predictor.translate(data_iter, model_file, cal_rouge=False, save=False, save_by_id=True)

    def plot_feat(self, test_loader, cache_dir, model_file=''):
        from analysis.plot_feat_tsne import PlotTSNE

        test = CommentDataset(self.args, test_loader, cache_dir, self.dataname, self.device, self.tokenizer)
        test_iter = Iterator(test, batch_size=len(test), device=self.device, shuffle=False) # 1651

        model_path = os.path.join(self.args.savepath, model_file)
        if os.path.exists(model_path):
            print('Plotting feature for {}'.format(model_file))
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)['model'], strict=False)

            plottsne = PlotTSNE(self.args, self.model)
            plottsne.plot(test_iter, model_file.split('_')[0])
        else:    
            best_model = os.path.join(self.args.savepath, 'best_model.pt')
            if os.path.exists(best_model):
                print('Ploting feature for best_model.pt')
                self.model.load_state_dict(torch.load(best_model, map_location=lambda storage, loc: storage)['model'], strict=False)

                plottsne = PlotTSNE(self.args, self.model)
                plottsne.plot(test_iter, 'best')

            for epoch in range(0, self.maxepoch, 1):
                model_path= os.path.join(self.args.savepath, '{}_model.pt'.format(epoch))
                if os.path.exists(model_path):
                    print('Plotting feature for epoch {}_model.pt'.format(epoch))
                    self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)['model'], strict=False)
                    plottsne = PlotTSNE(self.args, self.model)
                    plottsne.plot(test_iter, str(epoch))

    def inference_for_demo(self, loader, run_gen=False, portion='all', level=3):
        """ Testing detector  """
        test = TreeDataset(self.args, loader, self.dataname, self.device, self.tokenizer)

        data_iter = Iterator(test, train=False, device=self.device, 
                             batch_size=len(test) if len(test)<self.bs else self.bs,
                             sort_key=lambda x: len(x.src),
                             sort_within_batch=False)
        # Define trainer
        train_loss = abs_loss(self.args.label_num, self.maxepoch, self.device, train=False)
        trainer = build_trainer(self.args, self.model, self.optim, train_loss)

        if level == 1:
            logger.info('Test on detection model (stage-1)')
            best_model = os.path.join(self.args.savepath, 'best_model.pt')
            ckpt = torch.load(best_model, map_location=lambda storage, loc: storage)['model']
            try:
                self.model.load_state_dict(ckpt)
            except:
                mismatch = self.model.load_state_dict(ckpt, strict=False)
                print(mismatch)
                logger.info('[Warning] The keys in state dict do not strictly match')

            trainer.testing(data_iter, tokenizer=self.tokenizer, gen_flag=False, info="{}".format(best_model.split("/")[-1]), write_type="w")

            logger.info('Test on final model (stage-3)')
            best_model = os.path.join(self.args.savepath, 'best_final_model.pt')
            ckpt = torch.load(best_model, map_location=lambda storage, loc: storage)['model']
            try:
                self.model.load_state_dict(ckpt)
            except:
                mismatch = self.model.load_state_dict(ckpt, strict=False)
                print(mismatch)
                logger.info('[Warning] The keys in state dict do not strictly match')

            trainer.testing(data_iter, tokenizer=self.tokenizer, gen_flag=False, info="{}".format(best_model.split("/")[-1]), write_type="a")


        if level >= 2:
            logger.info('Test on adv model (stage-2)')
            best_model = os.path.join(self.args.savepath, 'best_adv_model.pt')
            ckpt = torch.load(best_model, map_location=lambda storage, loc: storage)['model']
            try:
                self.model.load_state_dict(ckpt)
            except:
                mismatch = self.model.load_state_dict(ckpt, strict=False)
                print(mismatch)
                logger.info('[Warning] The keys in state dict do not strictly match')

            # Test without generated response
            trainer.testing(data_iter, tokenizer=self.tokenizer, gen_flag=False, info="{}".format(best_model.split("/")[-1]), write_type="w")

            # Test with generated response
            _, wrongs_before = trainer.testing(data_iter, tokenizer=self.tokenizer, gen_flag=True, info="{} With Generated Response".format(best_model.split("/")[-1]), write_type="a", output_wrong_pred=True)
            predictor = build_predictor(self.args, self.model, self.tokenizer, self.symbols, logger)
            predictor.translate(data_iter, 'best', have_gold=False, info="{} With Generated Response".format(best_model.split("/")[-1]))


        if level >= 3:
            logger.info('Test on final model (stage-3)')
            best_model = os.path.join(self.args.savepath, 'best_model.pt')
            ckpt = torch.load(best_model, map_location=lambda storage, loc: storage)['model']
            #ckpt['classifier.pooler.conv.lin.weight'] = ckpt['classifier.pooler.conv.weight']
            try:
                self.model.load_state_dict(ckpt)
            except:
                mismatch = self.model.load_state_dict(ckpt, strict=False)
                print(mismatch)
                logger.info('[Warning] The keys in state dict do not strictly match')

            # Test without generated response
            trainer.testing(data_iter, tokenizer=self.tokenizer, gen_flag=False, info="{}".format(best_model.split("/")[-1]), write_type="a")

            # Test with generated response
            _, wrongs_after = trainer.testing(data_iter, tokenizer=self.tokenizer, gen_flag=True, info="{} With Generated Response".format(best_model.split("/")[-1]), write_type="a", output_wrong_pred=True)
            predictor = build_predictor(self.args, self.model, self.tokenizer, self.symbols, logger)
            predictor.translate(data_iter, 'best', have_gold=False, info="{} With Generated Response".format(best_model.split("/")[-1]))

            # Find the data that are successfully attack and fixed
            fixed = []
            for i in wrongs_before:
                if i not in wrongs_after:
                    fixed.append(str(i))

            wrongs_before = [str(i) for i in wrongs_before]
            with open(os.path.join(self.args.savepath, "id_fixed.txt"), "w") as f:
                f.write("\n".join(fixed))

            with open(os.path.join(self.args.savepath, "id_attack_success.txt"), "w") as f:
                f.write("\n".join(wrongs_before))

