"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from others.reporter import Statistics
from eval.evaluate import evaluationclass, evaluation4class, evaluation3class

def abs_loss(label_num, maxepoch, device, train=True,
             train_gen=False, generator=None, symbols=None, vocab_size=None, label_smoothing=0.0, sample_per_cls=None):
    compute = NMTLossCompute(
        label_num, maxepoch,
        train_gen, generator, symbols, vocab_size, label_smoothing=label_smoothing if train else 0.0, sample_per_cls=sample_per_cls)
    compute.to(device)
    return compute



class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, symbols):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        if symbols is not None:
            self.padding_idx = symbols['PAD']
        self.epoch = 0



    def _make_shard_state(self, batch, output,  attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, epoch, normalization=None):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        self.epoch = epoch
        shard_state = self._make_shard_state(batch, output)
        loss, batch_stats = self._compute_loss(normalization, **shard_state)

        return batch_stats, loss

    def sharded_compute_loss(self, batch, output,
                             shard_size,
                             epoch,
                             normalization=None):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        self.epoch = epoch
        # inital a statistic with all zeros
        batch_stats = Statistics()
        shard_state = self._make_shard_state(batch, output)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(normalization, **shard)
            loss.backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, loss_det, logits, label, loss_gen=None, scores=None, target=None):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        n_docs = len(logits)
        pred = logits.max(1)[1]
        #num_correct_det = pred.eq(label.view(-1)).sum().item()
        #results = evaluationclass(label.view(-1), label.view(-1))

        if self.label_num == 2:
            results = evaluationclass(pred, label.view(-1))
            results = (*results, 0, 0, 0, 0, 0, 0, 0, 0)


        elif self.label_num == 4:
            results = evaluation4class(pred, label.view(-1))

        elif self.label_num == 3:
            results = evaluation3class(pred, label.view(-1))

        if loss_gen is not None:
            pred = scores.max(1)[1]
            non_padding = target.ne(self.padding_idx)
            num_correct_token = pred.eq(target) \
                              .masked_select(non_padding) \
                              .sum() \
                              .item()
            num_non_padding = non_padding.sum().item()

            # build statistic for later update to other statistic
            return Statistics(loss.item(), loss_det.item(), *results, n_docs,
                              loss_gen.item(), num_non_padding, num_correct_token)
        return Statistics(loss.item(), loss_det.item(), *results, n_docs)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
        return F.kl_div(output, model_prob, reduction='sum') / len(target)


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, label_num, maxepoch,
                 train_gen=None, generator=None, symbols=None, vocab_size=None, label_smoothing=0.0, sample_per_cls=None):
        super(NMTLossCompute, self).__init__(generator, symbols)
        self.train_gen = train_gen
        if train_gen:
            self.sparse = not isinstance(generator[1], nn.LogSoftmax)
            if label_smoothing > 0:
                self.criterion = LabelSmoothingLoss(
                    label_smoothing, vocab_size, ignore_index=self.padding_idx
                )
            else:
                self.criterion = nn.NLLLoss(
                    ignore_index=self.padding_idx, reduction='sum'
                )

        self.label_num = label_num
        self.maxepoch = maxepoch

        # If consider class imbalance
        if sample_per_cls:
            beta = 0.999
            no_of_classes = np.sum(sample_per_cls)
            effective_num = 1.0 - np.power(beta, sample_per_cls)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = torch.tensor(weights / np.sum(weights)).float()
            #weights = weights / np.sum(weights) * no_of_classes 
            #weights = torch.tensor(weights).float()
            #weights = weights.unsqueeze(0)
            #weights = weights.repeat()

            self.xent = nn.CrossEntropyLoss(reduction='mean', weight=weights)
        else:
            self.xent = nn.CrossEntropyLoss(reduction='mean')

    def _make_shard_state(self, batch, output, normalization=None):
        try:
            target_gen = batch.tgt[:,1:]
            cal_gen_loss = True
        except:
            cal_gen_loss = False

        if len(output)>1 and cal_gen_loss:
            return {
                "logits": output[0],
                "label" : batch.label,
                "output": output[1],
                "target": batch.tgt[:,1:],
                #"normalization": normalization,
            }
        else:
            return {
                "logits": output[0],
                "label" : batch.label,
            }

    def _compute_loss(self, normalization, logits, label, output=None, target=None):

        if self.train_gen and target is not None:
            # Prepare materials
            bottled_output = self._bottle(output) # reshape to 2-dim
            scores = self.generator(bottled_output)
            gtruth = target.contiguous().view(-1)

            # Calculate losses
            if normalization is None:
                normalization = 1
            #loss_gen = self.criterion(scores, gtruth).div(float(normalization))
            loss_gen = self.criterion(scores, gtruth)
            loss_det = self.xent(logits.view(-1, self.label_num), label.view(-1))

            #print('LossGen {:.4f} LossDet {:.4f}'.format(loss_gen.item(), loss_det.item()))
            # Adjust the loss weighting by epoch
            #weight = max(10-self.epoch, 1)
            #loss = (1-weight_det) * loss_gen + weight_det * loss_det
            loss = loss_gen + loss_det
            stats = self._stats(loss.clone(), loss_det.clone(), logits, label, loss_gen.clone(), scores, gtruth)


        else:
            # Only train detector
            loss = self.xent(logits.view(-1, self.label_num), label.view(-1))
            #print('LossDet {:.4f}'.format(loss.item()))
            stats = self._stats(loss.clone(), loss.clone(), logits, label)

        return loss, stats
        '''
        if self.train_detector_loss:
            # Prepare materials
            bottled_output = self._bottle(output) # reshape to 2-dim
            scores = self.generator(bottled_output)
            gtruth = target.contiguous().view(-1)

            # Calculate loss
            loss = self.criterion(scores, gtruth)
            loss_det = self.xent(logits.view(-1, self.num_labels), label.view(-1))

            # Adjust the loss weighting by epoch
            weight_det = min(self.epoch/(self.maxepoch -10), 1.0)

            loss = (1-weight_det) * loss + weight_det * loss_det
            stats = self._stats(loss.clone(), scores, gtruth, loss_det.clone(), logits, label)

        else:
            bottled_output = self._bottle(output) # reshape to 2-dim
            scores = self.generator(bottled_output)
            gtruth = target.contiguous().view(-1)

            loss = self.criterion(scores, gtruth)
            stats = self._stats(loss.clone(), scores, gtruth)
        '''

        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))
       
        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
