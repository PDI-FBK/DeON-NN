import os


class Main(object):
    """docstring for Rnn"""
    def __init__(self, config):
        self.config = config
        self.sess = None

    def run(self, basedir, force):
        dataset_path = os.path.join(basedir, self.config['dataset'])

        return
