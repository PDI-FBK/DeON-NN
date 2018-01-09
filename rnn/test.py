from rnn.model import Model
import os
import tensorflow as tf


class Test(Model):

    def __init__(self, config):

        self._max = 0

        with tf.Graph().as_default(), tf.device(config.test_device):
            super().__init__(config.checkpoint_dir)
            self.summary_path = os.path.join(config.summaries, "test")
            self.build(config)

    def step(self):
        res = self.sess.run([
            self.accuracy,
            self.tvars,
            self.summary_op,
            self.global_step])
        return res
