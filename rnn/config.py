import os
import logging


class Config():

    def __init__(self, config, basedir):
        self.emb_dim = config['emb_dim']
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']

        self.validate_every_steps = config['validate_every_steps']
        self.epochs = config['epochs']

        self.vocab_input_size = config['vocab_input_size']
        self.vocab_ouput_size = config['vocab_ouput_size']

        self.seed = config['seed']
        self.cell_type = config['cell_type']


        if not os.path.exists(config['saver_result_path']):
            raise Exception(config['saver_result_path'] + 'does not exist! Please use a valid path.')
        if not os.path.exists(config['saver_result_path']):
            os.makedirs(os.path.join(config['saver_result_path'], config['name']))

        self.logger = self._logging_setup(
            os.path.join(config['saver_result_path'], config['name'] + '.log'))
        self.summaries = os.path.join(
            basedir,
            config['saver_result_path'] + '/' + config['name'] + '/' + config['summaries'])
        self.checkpoint_dir = os.path.join(
            basedir,
            config['saver_result_path'] + '/' + config['name'] + '/' + config['checkpoint'])
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

    class Train():

        def __init__(self, params, basedir):
            self.inputfile = os.path.join(basedir, params['file'])
            self.device = params["device"]
            self.batch_size = params["batch_size"]

    class Test():

        def __init__(self, params, basedir):
            self.inputfile = os.path.join(basedir, params['file'])
            self.device = params["device"]
            self.batch_size = params["batch_size"]

    class Validation():

        def __init__(self, params, basedir):
            self.inputfile = os.path.join(basedir, params['file'])
            self.device = params["device"]
            self.batch_size = params["batch_size"]
