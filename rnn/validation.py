from rnn.model import Model
import os
import tensorflow as tf


class Validation(Model):

    def __init__(self, config):

        self.total_accuracy = 0
        self.count_accuracy = 0
        self._step = 0
        self.max_steps = config.validation.steps

        with tf.Graph().as_default(), tf.device(config.validation.device):
            super().__init__(config.checkpoint_dir)
            self.MODE = 'VALIDATION'
            self.summary_path = os.path.join(config.summaries, "validation")
            self.logger = config.logger
            self.keep_prob_vl = config.validation.keep_prob
            self.build(config, config.validation.inputfile, config.validation.batch_size)

    def step(self):
        res = self.sess.run([
            self.summary_op,
            self.accuracy,
            self.global_step], feed_dict={self.keep_prob: self.keep_prob_vl})
        self._step = res[2]
        self.total_accuracy += res[1]
        self.count_accuracy += 1
        return res

    def mean_accuracy(self):
        if self.count_accuracy > 0:
            return self.total_accuracy / self.count_accuracy
        return 0

    def summarize(self):
        _mean_accuracy = self.mean_accuracy()
        self.logger.info('Validation: step={} with mean_accuracy={}'.format(self._step, _mean_accuracy))
        metric = {'accuracy': _mean_accuracy}
        self.summary_writer.add_summary(
            self.as_summary(metric),
            global_step=self._step)
        self.summary_writer.flush()
