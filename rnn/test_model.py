from rnn.model import Model
import os
import tensorflow as tf
import rnn.common as common


class Config():

    def __init__(self, steps, inputfile, summary_path, mode, logger, keep_prob, device, checkpoint_dir, batch_size):
        self.steps = steps
        self.inputfile = inputfile
        self.summary_path = summary_path
        self.mode = mode
        self.logger = logger
        self.keep_prob = keep_prob
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size

class TestModel(Model):

    def __init__(self, global_config, mode_config):

        self.total_accuracy = 0
        self.total_precision = 0
        self.total_recall = 0
        self.count = 0
        self._step = 0

        with tf.Graph().as_default(), tf.device(mode_config.device):
            super().__init__(mode_config.checkpoint_dir)
            self.max_steps = mode_config.steps
            self.inputfile = mode_config.inputfile
            self.summary_path = mode_config.summary_path
            self.MODE = mode_config.mode
            self.logger = mode_config.logger
            self.keep_prob_vl = mode_config.keep_prob
            self.build(global_config, mode_config.batch_size)

    def step(self):
        res = self.sess.run([
            self.summary_op,
            self.accuracy,
            self.precision,
            self.recall,
            self.global_step], feed_dict={self.keep_prob: self.keep_prob_vl})
        self.logger.info('{} precision={}, recall={}'.format(self.MODE, res[2], res[3]))
        self._step = res[-1]
        self.total_accuracy += res[1]
        self.total_precision += res[2][0]
        self.total_recall += res[3][0]
        self.count += 1
        return res

    def mean_accuracy(self):
        if self.count > 0:
            return self.total_accuracy / self.count
        return 0

    def mean_recall(self):
        if self.count > 0:
            return self.total_recall / self.count
        return 0

    def mean_precision(self):
        if self.count > 0:
            return self.total_precision / self.count
        return 0

    def summarize(self):
        _mean_accuracy = self.mean_accuracy()
        _mean_precision = self.mean_precision()
        _mean_recall = self.mean_recall()
        self.logger.info('Test: step={} with mean_accuracy={}'.format(self._step, _mean_accuracy))
        self.logger.info('Test: step={} with mean_precision={}'.format(self._step, _mean_precision))
        self.logger.info('Test: step={} with mean_recall={}'.format(self._step, _mean_recall))
        metric = {
            'accuracy': _mean_accuracy,
            'precision': _mean_precision,
            'recall': _mean_recall,
            'f1': common.f1(_mean_precision, _mean_recall)}
        self.summary_writer.add_summary(
            self.as_summary(metric),
            global_step=self._step)
        self.summary_writer.flush()


class Test(TestModel):

    def __init__(self, config):
        mode_config = Config(
            config.test.steps,
            config.test.inputfile,
            os.path.join(config.summaries, 'TEST'),
            'TEST',
            config.logger,
            config.test.keep_prob,
            config.test.device,
            config.checkpoint_dir,
            config.test.batch_size)
        super().__init__(config, mode_config)


class Validation(TestModel):

    def __init__(self, config):
        mode_config = Config(
            config.validation.steps,
            config.validation.inputfile,
            os.path.join(config.summaries, 'VALIDATION'),
            'VALIDATION',
            config.logger,
            config.validation.keep_prob,
            config.validation.device,
            config.checkpoint_dir,
            config.validation.batch_size)
        super().__init__(config, mode_config)
