""" Report manager utility """
from __future__ import print_function
from datetime import datetime

import time
import math
import sys

from others.distributed import all_gather_list
from others.logging import logger
import pdb

def build_report_manager(opt):
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(opt.tensorboard_log_dir
                               + datetime.now().strftime("/%b-%d_%H-%M-%S"),
                               comment="Unmt")
    else:
        writer = None

    report_mgr = ReportMgr(opt.report_every, start_time=-1,
                           tensorboard_writer=writer)
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats, multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if multigpu:
            report_stats = Statistics.all_gather_stats(report_stats)

        if step % self.report_every == 0:
            self._report_training(
                step, num_steps, learning_rate, report_stats)
            self.progress_step += 1
        return Statistics()

    def _report_training(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()

    def report_step(self, lr, step, train_stats=None, valid_stats=None, test_stats=None):
        """
        Report stats of a step

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self._report_step(
            lr, step, train_stats=train_stats, valid_stats=valid_stats, test_stats=test_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()


class ReportMgr(ReportMgrBase):
    def __init__(self, report_every, start_time=-1., writer=None, label_num=2):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(ReportMgr, self).__init__(report_every, start_time)
        self.writer = writer
        self.label_num = label_num

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, step_):
        if self.writer is not None:
            stats.log_tensorboard(
                prefix, learning_rate, step_, self.writer, self.label_num)

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        # Log the progress using the number of batches on the x-axis.
        self.maybe_log_tensorboard(report_stats,
                                   "progress",
                                   learning_rate,
                                   step)
        report_stats = Statistics()
        return report_stats

    def _report_step(self, lr, step, train_stats=None, valid_stats=None, test_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train detector xent: {:.4f}'.format(train_stats.det_xent()))
            self.log('Train detector acc: {:.4f}'.format(train_stats.det_acc()))
            self.log('Train detector F1: {:.4f} {:.4f}'.format(train_stats.det_f1(),
                                                               train_stats.det_f2()))
            if train_stats.gen_acc()>0:
                self.log('Train generator xent: {:4f}'.format(train_stats.gen_xent()))
                self.log('Train generator acc: {:.4f}'.format(train_stats.gen_acc()))

            self.maybe_log_tensorboard(train_stats,"train",lr,step)

        if valid_stats is not None:
            self.log('Epoch: {}'.format(step))
            self.log('Valid detector loss:  {:.4f}'.format(valid_stats.total_loss()))
            self.log('Valid detector xent:  {:.4f}; {}'
              .format(valid_stats.det_xent(), valid_stats.get_simple_message(self.label_num)))

            if valid_stats.gen_acc()>0:
                self.log('Valid generator xent: {:.4f}; acc: {:.4f}'
                  .format(valid_stats.gen_xent(), valid_stats.gen_acc()))

            self.maybe_log_tensorboard(valid_stats,"valid",lr,step)

        if test_stats is not None:
            self.log('Test detector loss:  {:.4f}'.format(test_stats.total_loss()))
            self.log('Test detector xent:  {:.4f}; {}'
                .format(test_stats.det_xent(), test_stats.get_simple_message(self.label_num)))

            if test_stats.gen_acc()>0:
                self.log('Test generator xent: {:.4f}; acc: {:.4f}'
                  .format(test_stats.gen_xent(), test_stats.gen_acc()))

            self.maybe_log_tensorboard(test_stats,"test",lr,step)

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, loss_det=0, acc=0,
                 a1=0, p1=0, r1=0, f1=0, 
                 a2=0, p2=0, r2=0, f2=0,
                 a3=0, p3=0, r3=0, f3=0, 
                 a4=0, p4=0, r4=0, f4=0,
                 n_docs=0, loss_gen=0, n_words=1, n_correct=0):

        self.loss = loss
        self.loss_gen = loss_gen
        self.n_words = n_words
        self.n_docs = n_docs
        self.n_correct = n_correct
        self.n_src_words = 0
        self.loss_det = loss_det
        self.acc = acc
        self.a1 = a1
        self.p1 = p1
        self.r1 = r1
        self.f1 = f1
        self.a2 = a2
        self.p2 = p2
        self.r2 = r2
        self.f2 = f2
        self.a3 = a3
        self.p3 = p3
        self.r3 = r3
        self.f3 = f3
        self.a4 = a4
        self.p4 = p4
        self.r4 = r4
        self.f4 = f4
        self.n_update = 0
        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        from torch.distributed import get_rank

        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.loss_gen += stat.loss_gen
        self.loss_det += stat.loss_det
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_docs += stat.n_docs
        self.acc += stat.acc
        self.a1 += stat.a1
        self.p1 += stat.p1
        self.r1 += stat.r1
        self.f1 += stat.f1
        self.a2 += stat.a2
        self.p2 += stat.p2
        self.r2 += stat.r2
        self.f2 += stat.f2
        self.a3 += stat.a3
        self.p3 += stat.p3
        self.r3 += stat.r3
        self.f3 += stat.f3
        self.a4 += stat.a4
        self.p4 += stat.p4
        self.r4 += stat.r4
        self.f4 += stat.f4
        self.n_update += 1

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def total_loss(self):
        return self.loss / self.n_update

    def gen_acc(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def gen_xent(self):
        """ compute cross entropy """
        #return self.loss / self.n_words
        return self.loss_gen / self.n_update

    def det_xent(self):
        """ compute cross entropy of detector loss """
        #return self.loss_det / self.n_docs
        return self.loss_det / self.n_update

    def avg(self, accum_value):
        return accum_value / self.n_update

    def det_acc(self):
        return self.acc / self.n_update

    def det_p1(self):
        return self.p1 / self.n_update

    def det_r1(self):
        return self.r1 / self.n_update

    def det_f1(self):
        return self.f1 / self.n_update

    def det_p2(self):
        return self.p2 / self.n_update

    def det_r2(self):
        return self.r2 / self.n_update

    def det_f2(self):
        return self.f2 / self.n_update

    def det_f3(self):
        return self.f3 / self.n_update

    def det_f4(self):
        return self.f4 / self.n_update

    def gen_ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss_gen / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        logger.info(
            "Step{:2d}/{:5d}; det_xent:{:.4f}; acc:{:.2f}%; f1:{:.2f}%; lr:{:.8f}; {} sec"
            .format(step, num_steps, self.det_xent(), self.det_acc(), self.det_f1,
                   learning_rate, time.time()-start))
        if self.loss_gen!=0:
            logger.info(
                ("Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
                 "lr: %7.8f; %3.0f/%3.0f tok/s; %6.0f sec")
                % (step, num_steps,
                   self.gen_acc(),
                   self.gen_ppl(),
                   self.gen_xent(),
                   learning_rate,
                   self.n_src_words / (t + 1e-5),
                   self.n_words / (t + 1e-5),
                   time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, learning_rate, step, writer, label_num) :
        """ display statistics to tensorboard """
        t = self.elapsed_time()

        if writer is not None:
            if self.loss_gen!=0:
                writer.add_scalar(prefix+'/gen_xent', self.gen_xent(), step)
                #writer.add_scalar(prefix + "/gen_ppl",self.gen_ppl(), step)
                writer.add_scalar(prefix + "/gen_acc",self.gen_acc(), step)
                #writer.add_scalar(prefix + "/gen_tgtper",self.n_words/t, step)
            writer.add_scalar(prefix + "/lr",learning_rate, step)
            writer.add_scalar(prefix + "/det_xent",self.det_xent(), step)
            writer.add_scalar(prefix + "/det_acc",self.det_acc(), step)
            writer.add_scalar(prefix + "/loss",self.total_loss(), step)
            if label_num == 3:
                writer.add_scalar(prefix + "/det_f1",self.det_f1(), step)
                writer.add_scalar(prefix + "/det_f2",self.det_f2(), step)
                writer.add_scalar(prefix + "/det_f3",self.det_f2(), step)
                writer.add_scalar(prefix + "/det_f", (self.det_f1()+self.det_f2()+self.det_f3())/3, step)
            elif label_num == 2:
                writer.add_scalar(prefix + "/det_p1",self.det_p1(), step)
                writer.add_scalar(prefix + "/det_r1",self.det_r1(), step)
                writer.add_scalar(prefix + "/det_f1",self.det_f1(), step)
                writer.add_scalar(prefix + "/det_p2",self.det_p2(), step)
                writer.add_scalar(prefix + "/det_r2",self.det_r2(), step)
                writer.add_scalar(prefix + "/det_f2",self.det_f2(), step)
            elif label_num == 4:
                writer.add_scalar(prefix + "/det_f1",self.det_f1(), step)
                writer.add_scalar(prefix + "/det_f2",self.det_f2(), step)
                writer.add_scalar(prefix + "/det_f3",self.det_f2(), step)
                writer.add_scalar(prefix + "/det_f4",self.det_f4(), step)
                writer.add_scalar(prefix + "/det_f", (self.det_f1()+self.det_f2()+self.det_f3()+self.det_f4())/4, step)


    def get_simple_message(self, label_num):
        if label_num == 2:
            message = '{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                self.det_acc(),
                self.det_p1(), self.det_r1(), self.det_f1(),
                self.det_p2(), self.det_r2(), self.det_f2(),
            )

        elif label_num == 4:
            message = '{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},'.format(
                self.avg(self.acc),
                self.avg(self.f1), self.avg(self.f2),
                self.avg(self.f3), self.avg(self.f4),
            )
        elif label_num == 3:
            message = '{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                self.avg((self.f1+self.f2+self.f3)/3),
                self.avg(self.f1), self.avg(self.f2), self.avg(self.f3))

        return message

    def get_complete_message(self, label_num):
        if label_num == 2:
            message = '{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                self.det_acc(),
                self.det_p1(), self.det_r1(), self.det_f1(),
                self.det_p2(), self.det_r2(), self.det_f2(),
            )

        elif label_num == 4:
            message = '{:.4f},\
                    {:.4f},{:.4f},{:.4f},{:.4f},\
                    {:.4f},{:.4f},{:.4f},{:.4f},\
                    {:.4f},{:.4f},{:.4f},{:.4f},\
                    {:.4f},{:.4f},{:.4f},{:.4f}'.format(
                self.avg(self.acc),
                self.avg(self.a1), self.avg(self.p1), self.avg(self.r1), self.avg(self.f1),
                self.avg(self.a2), self.avg(self.p2), self.avg(self.r2), self.avg(self.f2),
                self.avg(self.a3), self.avg(self.p3), self.avg(self.r3), self.avg(self.f3),
                self.avg(self.a4), self.avg(self.p4), self.avg(self.r4), self.avg(self.f4),
            )
        elif label_num == 3:
            message = '{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                self.avg((self.f1+self.f2+self.f3)/3),
                self.avg(self.f1), self.avg(self.f2), self.avg(self.f3))



        return message

    def write_results(self, filedir, epoch=-1, label_num=2):
        output = '[{}],{}\n'.format(
            epoch, self.get_complete_message(label_num),
        )

        with open(filedir, 'a') as f:
            f.write(output)

