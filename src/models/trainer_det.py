import os, sys
sys.path.append(os.getcwd())
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from others.earlystopping2class import EarlyStopping
from eval.evaluate import *

from models.model_de import RumorDetector_BiGCN
from data.utils import WordEncoder

from tqdm import tqdm
import pdb


class RumorDetector(object):
    def __init__(self, args, model_savepath, vocab_vectors=None, iter=0, fold=0, wandb=None, root_enhance=True):
        super(RumorDetector, self).__init__()
        self.model_savepath = model_savepath
        self.iter = iter
        self.fold = fold
        #self.result_dir = cfg.SAVE_DIR
        input_dim = args.det_input_dim
        hid_dim = args.det_hid_dim
        out_dim = args.det_out_dim
        self.device = 'cuda:{}'.format(args.visible_gpus)
        self.maxepoch = args.det_max_epoch

        #self.use_text = cfg.USE_TEXT
        #if self.use_text:
        #    self.encoder = WordEncoder(cfg, vocab_vectors)

        self.model = RumorDetector_BiGCN(args, input_dim, hid_dim, out_dim, vocab_vectors, \
                device=self.device, root_enhance=root_enhance).to(self.device)
        self.ES = EarlyStopping(patience=args.det_patience, verbose=True)

        BU_params = list(map(id, self.model.BUrumorGCN.conv1.parameters()))
        BU_params+= list(map(id, self.model.BUrumorGCN.conv2.parameters()))
        base_params = filter(lambda p:id(p) not in BU_params, self.model.parameters())
        self.optimizer = optim.Adam([
            {'params':base_params},
            {'params':self.model.BUrumorGCN.conv1.parameters(), 'lr':args.det_lr/5},
            {'params':self.model.BUrumorGCN.conv2.parameters(), 'lr':args.det_lr/5}
            ], lr=args.det_lr, weight_decay=args.det_weight_decay)

        self.wandb = wandb
        #if wandb is not None:
        #    self.wandb.watch(self.model)

    def train(self, train_loader, test_loader):

        states = {}
        c_time = time.time()
        for epoch in range(self.maxepoch):

            train_loss, train_acc = self._train(train_loader, epoch)
            print('[Detector] Iter{:03d} | Fold{:03d} | Epoch {:03d} | TrainLoss {:.4f} | TrainAcc {:.4f}'\
                .format(self.iter, self.fold, epoch, train_loss, train_acc))

            loss, acc, acc1, pre1, rec1, f1, acc2, pre2, rec2, f2 = self.test(test_loader)
            print('[Detector] Iter{:03d} | Fold{:03d} | Epoch {:03d} | Loss {:.4f} | Acc {:.4f} | F1 {:.4f} | F2 {:.4f}'\
                .format(self.iter, self.fold, epoch, loss, acc, f1, f2))

            self.ES(-acc, acc, acc1, pre1, rec1, f1, acc2, pre2, rec2, f2,\
                epoch, self.model, self.model_savepath+'/detector.pkl')

            states.update({
                'detector/train_loss':train_loss,
                'detector/train_acc' :train_acc,
                'detector/test_loss' :loss,
                'detector/test_acc'  :acc,
                'detector/test_pre1' :pre1,
                'detector/test_rec1' :rec1,
                'detector/test_f1'   :f1,
                'detector/test_pre2' :pre2,
                'detector/test_rec2' :rec2,
                'detector/test_f2'   :f2,
                })

            #if self.wandb is not None:                
            #    for key, value in states.items():
            #        self.wandb.log({key:value}, epoch)

            if self.ES.early_stop or epoch==self.maxepoch-1:
                stop_epoch = self.ES.epoch
                acc = self.ES.acc
                acc1 = self.ES.acc1
                pre1 = self.ES.pre1
                rec1 = self.ES.rec1
                f1   = self.ES.f1
                acc2 = self.ES.acc2
                pre2 = self.ES.pre2
                rec2 = self.ES.rec2
                f2 = self.ES.f2
                cost_time = (time.time()-c_time)/60
                status = 'Detector'
                if 'victim' in self.model_savepath:
                    status = 'Victim'
                with open(self.model_savepath+'/average_performance.tsv', 'a') as f:
                    f.write('[{}]\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{:.2f}\n'\
                    .format(status, self.iter, self.fold, acc, acc1, pre1, rec1, f1, \
                    acc2, pre2, rec2, f2, stop_epoch, cost_time))
                return 


    def _train(self, loader, epoch):
        
        sum_acc, sum_loss, cnt_batch = 0, 0, 1
        self.model.train()
        for data in tqdm(loader):
            data.to(self.device)
            #data.x = self.encoder(data.x)
            pred = self.model(data)
            loss = F.nll_loss(F.log_softmax(pred, dim=1), data.y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _, pred = pred.max(dim=-1)
            correct = pred.eq(data.y).sum().item()
            acc = correct/len(data.y)

            sum_acc += acc
            sum_loss += loss.item()
            cnt_batch += 1
            #print('[Detector] Epoch{:03d | Batch{:03d}} TrainLoss{:.4f} TrainAcc{:.4f}'\
            #        .format(epoch, cnt_batch, loss.item(), acc))

        return sum_loss/len(loader), sum_acc/len(loader)

    def test(self, loader):
        sum_loss = 0
        result_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.model.eval()
        for data in tqdm(loader):
            data.to(self.device)
            pred = self.model(data)
            loss = F.nll_loss(F.log_softmax(pred, dim=1), data.y)

            sum_loss += loss.item()

            _, pred = pred.max(dim=1)
            results = evaluationclass(pred, data.y)
            
            for i in range(len(results)):
                result_list[i] += results[i]

        loss = sum_loss/len(loader)
        acc  = result_list[0]/len(loader)
        acc1 = result_list[1]/len(loader)
        pre1 = result_list[2]/len(loader)
        rec1 = result_list[3]/len(loader)
        f1   = result_list[4]/len(loader)
        acc2 = result_list[5]/len(loader)
        pre2 = result_list[6]/len(loader)
        rec2 = result_list[7]/len(loader)
        f2   = result_list[8]/len(loader)

        return loss, acc, acc1, pre1, rec1, f1, acc2, pre2, rec2, f2

    def inference_seq(self, loader, eids):
        #all_pred = {}
        cnt = 0
        if not os.path.exists('{}/prediction'.format(self.model_savepath)):
            os.makedirs('{}/prediction'.format(self.model_savepath))

        for data in tqdm(loader):
            data.to(self.device)

            eid = eids[cnt]
            cnt += 1
            pred_list = []
            for i in range(len(data.x)):
                
                x = data.x[:i+1]
                sub_edge = (data.edge_index[0]<i+1) * (data.edge_index[1]<i+1)
                edge = data.edge_index[:,sub_edge]
                BU_edge = data.BU_edge_index[:,sub_edge]
                num_edge = torch.tensor(edge.shape[1]).to(self.device)

                sub_graph = data.clone()
                sub_graph.x = x
                sub_graph.edge_index = edge
                sub_graph.BU_edge_index = BU_edge
                sub_graph.depth = data.depth[:edge.shape[1]]
                sub_graph.batch = data.batch[:i+1]

                pred = F.softmax(self.model(sub_graph), dim=1)

                pred_list.append(pred)
                #all_pred[eid][i] = pred

            all_pred = torch.cat(pred_list)
            difference = (all_pred[1:,:] - all_pred[:-1,:]).cpu().data

            torch.save(difference, '{}/prediction/{}.pt'.format(self.model_savepath,eid))
            #all_pred[eid] = torch.cat(pred_list).cpu()
        return 
        #return all_pred

    #def inference_eli(self, )

    def forward(self, data):
        return self.model(data)



