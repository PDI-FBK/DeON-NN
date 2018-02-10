from rnn.test import Test
import os


class Validation(Test):

    def __init__(self, config):

        super().__init__(config)
        self.MODE = 'VALIDATION'
        self.summary_path = os.path.join(config.summaries, "validation")
