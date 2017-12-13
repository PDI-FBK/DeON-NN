from rnn.model import Model
from rnn.model import ModelType
import tensorflow as tf


class Main(object):
    """docstring for Rnn"""
    def __init__(self, config):
        self.config = config

        self.training_steps = config.training_steps
        self.model_checkpoint = config.model_checkpoint
        with tf.Graph().as_default() as graph:
            self.train_model = Model(ModelType.TRAIN, config, graph)

    def run(self, force):
        for step in self.train_model.next():
            if step % self.training_steps == 0:
                print(step)
                self.train_model.save_checkpoint(self.model_checkpoint, step)
                self._run_all_test_model()

    def _run_all_test_model(self):
        with tf.Graph().as_default() as graph:
            test_model = Model(ModelType.TEST, self.config, graph)
            for step in test_model.next():
                print(step)
