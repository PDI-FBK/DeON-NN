from rnn.model import Model
from rnn.logits import Logits
import rnn.data as data
import os
import tensorflow as tf


class Train(Model):

    def __init__(self, config):
        with tf.Graph().as_default() as graph:
            super().__init__(config.checkpoint_dir)
            self.summary_path = os.path.join(config.summaries, 'train')

            tensor = self._get_tensor(config)
            y = tensor['definition']
            real_ouput = tf.cast(tf.reshape(y, [-1]), tf.float32)
            # self.graph = graph
            logits = Logits(config).build(tensor['words'], tensor['sentence_length'])
            self.build(real_ouput, logits)

    def step(self):
        res = self.sess.run([
            self.loss,
            self.accuracy,
            self.tvars,
            self.optimizer,
            self.summary_op,
            self.global_step])
        return res

    def _get_tensor(self, config):
        return data.inputs(
            [config.train_file],
            config.batch_size,
            True,
            1,
            config.seed)
