import os


class Config():

    def __init__(self, config, basedir):
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']

        self.summaries = os.path.join(basedir, config['summaries'])

        self.training_steps = config['training_steps']

        self.checkpoint_dir = os.path.join(basedir, config['checkpoint'])
        self.model_checkpoint = os.path.join(self.checkpoint_dir, 'MODEL')

        self.vocab_input_size = config['vocab_input_size']
        self.vocab_ouput_size = config['vocab_ouput_size']

        self.train_file = os.path.join(basedir, config['train.files'])
        self.test_file = os.path.join(basedir, config['test.files'])
        self.eval_file = os.path.join(basedir, config['eval.files'])

        self.seed = config['seed']
        self.cell_type = config['cell_type']
