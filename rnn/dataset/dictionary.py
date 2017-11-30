class Dictionary():

    def __init__(self, file_path):
        self._counter = 0
        self.vocab_index = dict()
        self.vocab_word = dict()
        self.file_path = file_path

    def get_data(self):
        ls = []
        for line in self.read_file():
            _id, _content, _, _, _def, _source = line.split('\t')
            words = _content.split()
            self._update_vocab(words)

            _def = int(_def.strip())
            ls.append((self._translate(words), _def))
        return ls

    def vocab_size(self):
        return self._counter

    def eos(self):
        return self.vocab_index['.']

    def read_file(self):
        with open(self.file_path, 'r') as f:
            for line in f:
                yield line

    def _update_vocab(self, words):
        for word in words:
            if word in self.vocab_index:
                continue
            self.vocab_index[word] = self._counter
            self.vocab_word[self._counter] = word
            self._counter += 1

    def _translate(self, words):
        return [self.vocab_index[word] for word in words]
