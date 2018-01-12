from rnn.model import Model
import os
import tensorflow as tf


class Train(Model):

    def __init__(self, config):
        with tf.Graph().as_default(), tf.device(config.train_device):
            super().__init__(config.checkpoint_dir)
            self.summary_path = os.path.join(config.summaries, 'train')
            self.logger = config.logger
            self.build(config)

    def step(self):
        res = self.sess.run([
            self.loss,
            self.accuracy,
            self.tvars,
            self.optimizer,
            self.summary_op,
            self.global_step])
        self.logger.info('Train loss={}, accuracy={}'.format(res[0], res[1]))
        self.logger.info('Train global_step={}'.format(res[-1]))
        return res
