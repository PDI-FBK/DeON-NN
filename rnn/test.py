from rnn.my_model import Model
from rnn.logits import Logits
import rnn.data as data
import os
import tensorflow as tf


class Test(Model):

    def __init__(self, config):
        super().__init__(config.checkpoint_dir)
        self.summary_path = os.path.join(config.summaries, 'train')

        tensor = self._get_tensor()
        y = tensor['definition']
        logits = Logits().build()
        self.build(y, logits)

    def step(self, merged):
        res = self.sess.run([
            self.accuracy,
            self.tvars,
            self.summary_op,
            self.global_step])
        return res

    def _get_tensor(self):
        return data.inputs(
            [self.config.train_file],
            self.config.batch_size,
            True,
            1,
            self.config.seed)
