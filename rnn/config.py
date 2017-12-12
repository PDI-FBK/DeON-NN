import os


class Config():

    def __init__(self, config, basedir):
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.hidden_size = config['hidden_size']
        self.training_steps = config['training_steps']
        self.summaries = os.path.join(basedir, config['summaries'])
        self.test_checkpoint = config['test_checkpoint']
        self.num_layers = config['num_layers']
        self.vocab_input_size = config['vocab_input_size']
        self.vocab_ouput_size = config['vocab_ouput_size']
        self.dataset_path = os.path.join(basedir, config['dataset'])
