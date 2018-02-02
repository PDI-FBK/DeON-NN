from rnn.model import Model
import os
import tensorflow as tf
import rnn.common as common


class Test(Model):

    def __init__(self, config):

        self.total_accuracy = 0
        self.total_precision = 0
        self.total_recall = 0
        self.count = 0
        self._step = 0
        self.max_steps = config.test.steps

        with tf.Graph().as_default(), tf.device(config.test.device):
            super().__init__(config.checkpoint_dir)
            self.MODE = 'TEST'
            self.summary_path = os.path.join(config.summaries, "test")
            self.logger = config.logger
            self.keep_prob_vl = config.test.keep_prob
            self.build(config, config.test.inputfile, config.test.batch_size)

    def step(self):
        res = self.sess.run([
            self.summary_op,
            self.accuracy,
            self.precision,
            self.recall,
            self.global_step], feed_dict={self.keep_prob: self.keep_prob_vl})
        self._step = res[2]
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
        metric = {
            'accuracy': _mean_accuracy,
            'precision': _mean_precision,
            'recall': _mean_recall,
            'f1': common.f1(_mean_precision, _mean_recall)}
        self.summary_writer.add_summary(
            self.as_summary(metric),
            global_step=self._step)
        self.summary_writer.flush()
