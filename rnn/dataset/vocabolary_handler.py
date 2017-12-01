from rnn.dataset.vocabolary import Vocabolary
import rnn.dataset.util as util
from pathlib import Path
import os

class VocabolaryHandler():

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_vocabolary(self):
        if self._vocabolary_path().exists():
            return self._load()
        return self._create(util._get_tsv_files(self.dataset_path))

    def _load(self):
        vocab_path = self._vocabolary_path()
        vocabolary = Vocabolary()
        with open(vocab_path, 'r') as f:
            for word in f:
                word = word.replace('\n', '')
                if word:
                    vocabolary.add(word)
        return vocabolary

    def _create(self, f_inputs):
        vocab_path = self._vocabolary_path()
        vocabolary = Vocabolary()
        vocabolary.add('<EOS>')
        for f_input in f_inputs:
            f_input_path = os.path.join(self.dataset_path, f_input)
            with open(f_input_path, 'r') as f:
                for line in f:
                    _, sentence, _, _, _, _ = line.split('\t')
                    for word in sentence.split():
                        vocabolary.add(word)
        with open(vocab_path, 'w') as f:
            for _, word in vocabolary.items():
                f.write(word + '\n')
        return vocabolary

    def _vocabolary_path(self):
        vocab_path = os.path.join(self.dataset_path, 'vocabolary.idx')
        return Path(vocab_path)


