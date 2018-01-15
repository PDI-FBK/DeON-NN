from rnn.model import Model
import os
import tensorflow as tf


class Validation(Model):

    def __init__(self, config):

        self.total_accuracy = 0
        self.count_accuracy = 0

        with tf.Graph().as_default(), tf.device(config.test_device):
            super().__init__(config.checkpoint_dir)
            self.MODE = 'VALIDATION'
            self.summary_path = os.path.join(config.summaries, "validation")
            self.logger = config.logger
            self.build(config, config.eval_file)

    def step(self):
        res = self.sess.run([
            self.summary_op,
            self.accuracy])
        self.total_accuracy += res[1]
        self.count_accuracy += 1
        return res

    def mean_accuracy(self):
        if self.count_accuracy > 0:
            return self.total_accuracy / self.count_accuracy
        return 0
