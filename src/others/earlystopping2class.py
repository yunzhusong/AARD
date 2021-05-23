import numpy as np
import torch

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
        self.epoch = 0
        self.acc=0
        self.f1=0
        self.f2 = 0
        self.val_loss_min = np.Inf

    def __call__(self,val_loss,acc,acc1,pre1,rec1,F1,acc2,pre2,rec2,F2,epoch,model,modelname):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.acc = acc
            self.acc1=acc1
            self.acc2=acc2
            self.pre1=pre1
            self.pre2=pre2
            self.rec1=rec1
            self.rec2=rec2
            self.f1 = F1
            self.f2 = F2
            self.epoch = epoch
            if model is not None:
                self.save_checkpoint(val_loss, model,modelname)
        elif score <= self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST LOSS:{:.4f}| Accuracy: {:.4f}|acc1: {:.4f}|pre1: {:.4f}|rec1: {:.4f}|F1: {:.4f}|acc2: {:.4f}|pre2: {:.4f}|rec2: {:.4f}|F2: {:.4f}"
                      .format(-self.best_score,self.acc,self.acc1,self.pre1,self.rec1,self.f1,self.acc2,self.pre2,self.rec2,self.f2))
        else:
            self.best_score = score
            self.acc = acc
            self.acc1=acc1
            self.acc2=acc2
            self.pre1=pre1
            self.pre2=pre2
            self.rec1=rec1
            self.rec2=rec2
            self.f1 = F1
            self.f2 = F2
            self.epoch = epoch
            if model is not None:
                self.save_checkpoint(val_loss, model,modelname)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,modelname):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        print('---> Save the best model')
        torch.save(model.state_dict(),modelname)
        self.val_loss_min = val_loss
