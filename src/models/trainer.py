import os
from os.path import join as pjoin
import pdb

import numpy as np
import torch
from tensorboardX import SummaryWriter

from others import distributed
from others.reporter import ReportMgr, Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str

from tqdm import tqdm

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, model, optims, loss, wandb=None):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if int(args.visible_gpus) >= 0:
        #gpu_rank = int(args.gpu_ranks[device_id])
        gpu_rank = 1
        n_gpu = 1
    else:
        gpu_rank = 0
        n_gpu = 0
    # For saving exp results (tensorboard and wandb)
    writer = None
    if args.log_tensorboard:
        tensorboard_log_dir = args.savepath
        writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    report_manager = ReportMgr(args.report_every, start_time=-1,
                               writer=writer, wandb=wandb, label_num = args.label_num)

    trainer = Trainer(args, model, optims, loss, grad_accum_count, n_gpu, gpu_rank, report_manager)

    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,  args, model,  optims, loss,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None, wandb=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_epoch = args.save_checkpoint_epoch
        self.model = model
        self.optims = optims
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager # for writing result to log file

        self.loss = loss
        self.epoch = 0

        assert grad_accum_count > 0


    def train(self, train_iter, train_steps, message=''):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter
        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter: same as train_iter_fct, for valid data
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        self.model.train()
        self.epoch += 1
        step =  self.optims[0]._step + 1
        one_iter = len(train_iter)

        true_batchs = []
        accum = 0
        normalization = 0

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        reduce_counter = 0

        tqdm_ = tqdm(train_iter, desc=message)
        for i, batch in enumerate(tqdm_):
            #print("batch index {}, 0/1/2: {}/{}/{}\r".format(i, len(np.where(batch.y.numpy()==0)[0]),
            #                                              len(np.where(batch.y.numpy()==1)[0]),
            #                                              len(np.where(batch.y.numpy()==2)[0])), end='')

            true_batchs.append(batch)
            if self.args.train_gen:
                num_tokens = batch.tgt[:, 1:].ne(self.loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization=None
            accum += 1
            if accum == self.grad_accum_count:
                reduce_counter += 1
                if self.n_gpu > 1 and normalization is not None:
                        normalization = sum(distributed
                                            .all_gather_list
                                            (normalization))

                self._gradient_accumulation(
                    true_batchs, normalization, total_stats,
                    report_stats)
                report_stats = self._maybe_report_training(
                    step, train_steps,
                    self.optims[0].learning_rate,
                    report_stats)

                true_batchs = []
                accum = 0
                normalization = 0
                step += 1
        #if ((self.epoch+1) % self.save_checkpoint_epoch == 0): #and self.gpu_rank == 0):
        #    self._save(self.epoch)

        return total_stats

    def validate(self, valid_iter, epoch=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()
        losses = []

        with torch.no_grad():
            tqdm_ = tqdm(valid_iter, desc='validating {}'.format(epoch))
            for batch in tqdm_:
                src = batch.src
                segs = batch.segs
                mask_src = batch.mask_src
                edges = batch.edges
                node_batch = batch.node_batch

                if self.args.train_gen:
                    tgt = batch.tgt
                    mask_tgt = batch.mask_tgt
                    outputs = self.model(src, segs, mask_src, edges, node_batch, tgt, mask_tgt)
                    batch_stats, loss = self.loss.monolithic_compute_loss(batch, outputs, self.epoch)
                else:
                    outputs = self.model(src, segs, mask_src, edges, node_batch)
                    batch_stats, loss = self.loss.monolithic_compute_loss(batch, outputs, self.epoch)

                stats.update(batch_stats)
                losses.append(loss.item())

            self._report_step(0, epoch, valid_stats=stats)
            print(sum(losses)/len(losses))

            return stats

    def testing(self, test_iter, step=0, gen_flag=False, tokenizer=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        from collections import defaultdict
        prediction_dict = defaultdict(list)
        label_dict = defaultdict(list)

        with torch.no_grad():
            for batch in test_iter:
                src = batch.src
                segs = batch.segs
                mask_src = batch.mask_src
                edges = batch.edges
                node_batch = batch.node_batch

                outputs = self.model(src, segs, mask_src, edges, node_batch, gen_flag=gen_flag)
                batch_stats, loss = self.loss.monolithic_compute_loss(batch, outputs, self.epoch)

                # write out prediction on different time interval
                if tokenizer:
                    predictions = outputs[0].max(axis=1)[1]
                    
                    for idx, id_ in enumerate(batch.id):
                        label = batch.label[idx].item()
                        prediction = predictions[idx]
                        num_node = len(node_batch[idx])
                        sent = ' '.join(tokenizer.convert_ids_to_tokens(src[idx][src[idx]!=0])).replace(' ##', '')
                        with open(pjoin(self.args.savepath, 'test_prediction.csv'), 'a') as f:
                            f.write('{},{},{},{}\n'.format(id_[0].item(), num_node, prediction, sent))
                        prediction_dict[id_[0].item()].append(prediction.item())
                        label_dict[id_[0].item()] = label

                stats.update(batch_stats)
            if tokenizer:
                with open(pjoin(self.args.savepath, 'alter_id_list.csv'), 'w') as f:
                    for id_, value in prediction_dict.items():
                        if len(set(value)) != 1:
                            f.write('{},{},{}\n'.format(id_, label_dict[id_], value))

            self._report_step(0, step, test_stats=stats)

            return stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            segs = batch.segs
            mask_src = batch.mask_src
            edges = batch.edges
            node_batch = batch.node_batch

            if self.args.train_gen:
                tgt = batch.tgt
                mask_tgt = batch.mask_tgt
                outputs = self.model(src, segs, mask_src, edges, node_batch, tgt, mask_tgt)
                batch_stats = self.loss.sharded_compute_loss(batch, outputs,
                    self.args.generator_shard_size, self.epoch, normalization)
            else:
                outputs = self.model(src, segs, mask_src, edges, node_batch)
                batch_stats = self.loss.sharded_compute_loss(batch, outputs,
                    self.args.generator_shard_size, self.epoch)

            #batch_stats.n_docs = int(src.size(0))

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))

                for o in self.optims:
                    o.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            for o in self.optims:
                o.step()


    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        
        #optims = [self.optims[0].optimizer.state_dict(), self.optims[1].optimizer.state_dict()]
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            #'optims': optims,
        }
        checkpoint_path = pjoin(self.args.savepath, '{}_model.pt'.format(step))
        logger.info("Saving checkpoint {}".format(checkpoint_path))
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None, test_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats, test_stats=test_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

    def exp_pos(self, test_iter, step=0):
        """ For running position experiment, not regular function 
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            diffs = []
            max_diff = 0
            num_wrong = 0
            for batch in test_iter:
                src = batch.src
                segs = batch.segs
                mask_src = batch.mask_src
                edges = batch.edges
                node_batch = batch.node_batch

                highest = 0
                lowest = 1
                wrong = False
                for node in range(len(node_batch[0])):
                    weight = torch.zeros((len(node_batch[0]))).to(src.device)
                    weight[node] = 1 
                    weight = [weight]
                    outputs = self.model.exp_pos(src, segs, mask_src, weight, edges, node_batch)
                    logit = outputs[0][0]
                    prob = torch.exp(logit)/(1+torch.exp(logit))
                    prob = prob[batch.label.reshape(-1)[0]]

                    # Calculate accuracy
                    if prob < 0.5:
                        wrong = True

                    # Find max and min under all possible possition
                    if prob > highest:
                        highest = prob
                    if prob < lowest:
                        lowest = prob

                if wrong:
                    num_wrong += 1

                print(highest, lowest)
                diff = highest - lowest
                diffs.append(diff)
                if diff > max_diff:
                    max_diff = diff

            error = num_wrong/len(test_iter)
            print(sum(diffs)/len(diffs))
            print(max_diff)
            print('{:.4f}'.format(error))
            with open(pjoin(self.args.savepath, 'exp_pos.txt'), 'a') as f:
                f.write('[test-pos],{:.4f},{:.4f},{:.4f}'.format(max_diff, diff, error))


    # NOTE: No Use
    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """
        Havn't modified by yunzhu !!!!!
             Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        can_path = '%s_step%d.candidate'%(self.args.result_path,step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        gold = []
                        pred = []
                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if(len(batch.src_str[i])==0):
                                continue
                            for j in selected_ids[i][:len(batch.src_str[i])]:
                                if(j>=len( batch.src_str[i])):
                                    continue
                                candidate = batch.src_str[i][j].strip()
                                _pred.append(candidate)

                                if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                    break

                            _pred = '<q>'.join(_pred)
                            if(self.args.recall_eval):
                                _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                            pred.append(_pred)
                            gold.append(batch.tgt_str[i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip()+'\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip()+'\n')
        if(step!=-1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)

        return stats
