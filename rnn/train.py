from rnn.model import Model
import os
import tensorflow as tf


class Train(Model):

    def __init__(self, config):
        with tf.Graph().as_default(), tf.device(config.train.device):
            super().__init__(config.checkpoint_dir)
            self.MODE = 'TRAIN'
            self.summary_path = os.path.join(config.summaries, 'train')
            self.logger = config.logger
            self.epochs = config.epochs
            self._step = 0
            self.build(config, config.train.inputfile, config.train.batch_size)

    def step(self):
        res = self.sess.run([
            self.summary_op,
            self.loss,
            self.accuracy,
            self.tvars,
            self.optimizer,
            self.global_step])
        self._step = res[-1]
        self.logger.info('Train loss={}, accuracy={}'.format(res[1], res[2]))
        self.logger.info('Train global_step={}'.format(res[-1]))

        metric = {'accuracy': res[2], 'loss': res[1]}
        self.summarize(metric)
        return res

    def flush(self):
        self.summary_writer.flush()

    def summarize(self, metric):
        self.summary_writer.add_summary(
            self.as_summary(metric),
            global_step=self._step)
        self.summary_writer.flush()
