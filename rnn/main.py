from rnn.train import Train
from rnn.test import Test
from rnn.validation import Validation
from datetime import datetime


class Main(object):
    """docstring for Rnn"""
    def __init__(self, config):
        self.config = config

        self.validate_every_steps = config.validate_every_steps
        self.model_checkpoint = config.model_checkpoint
        self.train_model = Train(config)
        self.logger = config.logger

    def run(self, force):
        self.logger.info('Start')
        for step in self.train_model.next():

            if step == 1 or step % 50 == 0:
                self.train_model.flush()
                self.train_model.save_checkpoint(self.model_checkpoint, step)

            if step % self.validate_every_steps == 0:
                self.logger.info('Saving checkpoint into {}'.format(self.model_checkpoint))
                self.train_model.save_checkpoint(self.model_checkpoint, step)
                self._run_test_model()
                self._run_validation_model()
        self._run_test_model()
        self._run_validation_model()

    def _run_test_model(self):
        self.logger.info('Run tests')
        model = Test(self.config)
        self._run_model(model, 'test')

    def _run_validation_model(self):
        self.logger.info('Run validation')
        model = Validation(self.config)
        self._run_model(model, 'validation')

    def _run_model(self, model, mode):
        start = datetime.now()
        for i, step in enumerate(model.next()):
            delta = datetime.now() - start
            combined = delta.seconds + delta.microseconds / 1E6
            self.logger.info('Single {} took: {} sec'.format(mode, combined))
            start = datetime.now()

            if i % 50 == 0:
                self.logger.info('After {} {}s accurasy={}'.format(i, mode, model.mean_accuracy()))

        if model.count_accuracy > 0:
            self.logger.info('{} mean_accuracy={}'
                .format(mode, model.mean_accuracy()))
