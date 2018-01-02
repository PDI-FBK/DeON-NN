from rnn.model import Model
from rnn.logits import Logits
import rnn.data as data
import os
import tensorflow as tf


class Test(Model):

    def __init__(self, config):
        with tf.Graph().as_default() as graph:
            super().__init__(config.checkpoint_dir)
            self.summary_path = os.path.join(config.summaries, 'test')
            self.device = 'CPU:0'

            tensor = self._get_tensor(config)
            y = tensor['definition']
            real_ouput = tf.cast(tf.reshape(y, [-1]), tf.float32)
            logits = Logits(config).build(tensor['words'], tensor['sentence_length'])
            self.build(real_ouput, logits)

    def step(self):
        res = self.sess.run([
            self.accuracy,
            self.tvars,
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
