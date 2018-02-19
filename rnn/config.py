import os
import logging
import tensorflow as tf
import glob


class Config():

    def __init__(self, config, basedir):
        self.emb_dim = config['emb_dim']
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']

        self.validate_every_steps = config['validate_every_steps']

        self.vocab_input_size = config['vocab_input_size']
        self.vocab_ouput_size = config['vocab_ouput_size']

        self.seed = config['seed']
        self.cell = Config.Cell(config['cell'], config['hidden_size'])
        self.optimizer = Config.Optimizer(config['optimizer']).optimizer

        if not os.path.exists(config['saver_result_path']):
            raise Exception(
                config['saver_result_path'] +
                'does not exist! Please use a valid path.')

        folder = os.path.join(
                basedir,
                config['saver_result_path'] + '/' + config['name'])
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.logger = self._logging_setup(
            os.path.join(folder, config['name'] + '.log'))
        self.summaries = os.path.join(folder, config['summaries'])
        self.checkpoint_dir = os.path.join(folder, config['checkpoint'])
        self.model_checkpoint = os.path.join(self.checkpoint_dir, 'MODEL')

        self.train = Config.Train(config["train"], basedir)
        self.test = Config.Test(config["test"], basedir)
        self.validation = Config.Validation(config["validation"], basedir)

    def _logging_setup(self, fpath):
        logger = logging.getLogger('deon-rnn')
        hdlr = logging.FileHandler(fpath, mode='a')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.DEBUG)
        return logger

    def __str__(self):
        return'TRAIN {} \nTEST {}\nVALIDATION{}'.format(
            self.train, self.test, self.validation)

    class Train():

        def __init__(self, params, basedir):
            self.inputfile = os.path.join(basedir, params['file.rio'])
            self.device = params["device"]
            self.batch_size = params["batch_size"]
            file_length = sum([Config.count_lines(f) for f in glob.glob(params['file.tsv'])])
            self.steps = params['epochs'] * (file_length // self.batch_size)
            self.keep_prob = params['keep_prob']

        def __str__(self):
            return 'device {}\tbatch size {}\tsteps {}'.format(
                self.device, self.batch_size, self.steps)

    class Test():

        def __init__(self, params, basedir):
            self.inputfile = os.path.join(basedir, params['file.rio'])
            self.device = params["device"]
            self.batch_size = params["batch_size"]
            self.steps = Config.one_epoch_steps(
                params['file.tsv'], self.batch_size)
            self.keep_prob = 1

        def __str__(self):
            return 'device {}\tbatch size {}\tsteps {}'.format(
                self.device, self.batch_size, self.steps)

    class Validation():

        def __init__(self, params, basedir):
            self.inputfile = os.path.join(basedir, params['file.rio'])
            self.device = params["device"]
            self.batch_size = params["batch_size"]
            self.steps = Config.one_epoch_steps(
                params['file.tsv'], self.batch_size)
            self.keep_prob = 1

        def __str__(self):
            return 'device {}\tbatch size {}\tsteps {}'.format(
                self.device, self.batch_size, self.steps)

    class Optimizer():

        def __init__(self, data):
            name = data['class']
            params = data['params']
            self.optimizer = tf.train.__dict__[name](**params)

    class Cell():

        def __init__(self, data, hidden_size):
            self.name = data['class']
            self.params = data['params']
            self.params['num_units'] = hidden_size

        def create(self):
            unresolve = 'could not resolve type `{}`'
            try:
                return tf.contrib.rnn.__dict__[self.name](**self.params)
            except:
                raise RuntimeError(unresolve.format(self.name))

    def count_lines(file):
        f_len = 0
        with open(file, 'r') as f:
            for _ in f:
                f_len += 1
        return f_len

    def one_epoch_steps(tsv_file, batch_size):
        f_len = Config.count_lines(tsv_file)
        return f_len // batch_size
