from rnn.model import Model
import os
import tensorflow as tf


class Test(Model):

    def __init__(self, config):

        self.total_accuracy = 0
        self.count_accuracy = 0
        self._step = 0
        self.max_steps = config.test.steps

        with tf.Graph().as_default(), tf.device(config.test.device):
            super().__init__(config.checkpoint_dir)
            self.MODE = 'TEST'
            self.summary_path = os.path.join(config.summaries, "test")
            self.logger = config.logger
            self.build(config, config.test.inputfile, config.test.batch_size)

    def step(self):
        res = self.sess.run([
            self.summary_op,
            self.accuracy,
            self.global_step])
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
        self.logger.info('Test: step={} with mean_accuracy={}'.format(self._step, _mean_accuracy))
        metric = {'accuracy': _mean_accuracy}
        self.summary_writer.add_summary(
            self.as_summary(metric),
            global_step=self._step)
        self.summary_writer.flush()
