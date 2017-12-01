import os.path
from rnn.dataset.vocabolary_handler import VocabolaryHandler

from rnn.dataset.data import *

class Main(object):
    """docstring for Rnn"""
    def __init__(self, config):
        self.config = config
        self.sess = None

    def run(self, basedir):
        dataset_path = os.path.join(basedir, self.config['dataset'])
        vocabolary = VocabolaryHandler(dataset_path).get_vocabolary()
        generate_rio_dataset(dataset_path, vocabolary)

        return



