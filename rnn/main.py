from rnn.train import Train
from rnn.test import Test


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
            if step % self.validate_every_steps == 0:
                self.logger.info('Saving checkpoint into {}'.format(self.model_checkpoint))
                self.train_model.save_checkpoint(self.model_checkpoint, step)
                self._run_all_test_model()
        self._run_all_test_model()

    def _run_all_test_model(self):
        self.logger.info('Run tests')
        test_model = Test(self.config)
        for step in test_model.next():
            pass
        if test_model.count_accuracy > 0:
            self.logger.info('Test mean_accuracy={}'
                .format(test_model.mean_accuracy))

