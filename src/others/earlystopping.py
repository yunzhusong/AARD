import numpy as np
import torch
import os
import wandb

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.acc=0
        self.f1=0
        self.f2 = 0
        self.f3 = 0
        self.f4 = 0
        self.epoch = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, acc, f1, f2, f3, f4, epoch, model, modelname):

        score = -val_loss
        #score = acc

        if self.best_score is None:
            self.best_score = score
            self.acc = acc
            self.f1 = f1
            self.f2 = f2
            self.f3 = f3
            self.f4 = f4
            self.epoch = epoch
            self.save_checkpoint(val_loss, model,modelname)
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.4f} | NR F1: {:.4f} | FR F1: {:.4f} | TR F1: {:.4f} | UR F1: {:.4f}"
                      .format(self.acc, self.f1, self.f2, self.f3, self.f4))
        else:
            self.best_score = score
            self.acc = acc
            self.f1 = f1
            self.f2 = f2
            self.f3 = f3
            self.f4 = f4
            self.epoch = epoch
            self.save_checkpoint(val_loss, model,modelname)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,modelname):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        print('---> Save the best detector model')
        torch.save(model.state_dict(), modelname)
        #torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
        self.val_loss_min = val_loss


class EarlyStoppingAttacker:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.acc=0
        self.accrec = 0
        self.accgen = 0
        self.sucess = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, acc, accrec, accgen, sucess, attacker, identifier, savename_at,
            savename_id, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.acc = acc
            self.accrec = accrec
            self.accgen = accgen
            self.sucess = sucess
            self.epoch = epoch
            self.save_checkpoint(val_loss, attacker, savename_at)
            self.save_checkpoint(val_loss, identifier, savename_id)
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.4f}|AccRec: {:.4f}|AccGen: {:.4f}|SucessRate: {:.4f}"
                      .format(self.acc, self.accrec, self.accgen, self.sucess))
        else:
            self.best_score = score
            self.acc = acc
            self.accrec = accrec
            self.accgen = accgen
            self.sucess = sucess
            self.epoch = epoch
            self.save_checkpoint(val_loss, attacker, savename_at)
            self.save_checkpoint(val_loss, identifier, savename_id)
            self.counter = 0

    
    def save_checkpoint(self, val_loss, model, savename):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        torch.save(model.state_dict(), savename)
        self.val_loss_min = val_loss

class EarlyStoppingFinetune:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.acc= 0
        self.f1 = 0
        self.f2 = 0
        self.f3 = 0
        self.f4 = 0
        self.acc_old=0
        self.f1_old =0
        self.f2_old = 0
        self.f3_old = 0
        self.f4_old = 0
        self.epoch = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, acc, f1, f2, f3, f4, acc_old, f1_old, f2_old, f3_old, f4_old, epoch, model, modelname):

        score = -val_loss
        #score = acc

        if self.best_score is None:
            self.best_score = score
            self.acc = acc
            self.f1 = f1
            self.f2 = f2
            self.f3 = f3
            self.f4 = f4
            self.acc_old = acc_old
            self.f1_old = f1_old
            self.f2_old = f2_old
            self.f3_old = f3_old
            self.f4_old = f4_old
            self.epoch = epoch
            self.save_checkpoint(val_loss, model,modelname)
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy new: {:.4f} | NR F1: {:.4f} | FR F1: {:.4f} | TR F1: {:.4f} | UR F1: {:.4f}"
                      .format(self.acc, self.f1, self.f2, self.f3, self.f4))
                print("BEST Accuracy old: {:.4f} | NR F1: {:.4f} | FR F1: {:.4f} | TR F1: {:.4f} | UR F1: {:.4f}"
                      .format(self.acc_old, self.f1_old, self.f2_old, self.f3_old, self.f4_old))
        else:
            self.best_score = score
            self.acc = acc
            self.f1 = f1
            self.f2 = f2
            self.f3 = f3
            self.f4 = f4
            self.acc_old = acc_old
            self.f1_old = f1_old
            self.f2_old = f2_old
            self.f3_old = f3_old
            self.f4_old = f4_old
            self.epoch = epoch
            self.save_checkpoint(val_loss, model,modelname)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,modelname):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        print('---> Save the best detector model')
        torch.save(model.state_dict(), modelname)
        self.val_loss_min = val_loss
