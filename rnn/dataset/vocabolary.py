class Vocabolary():

    def __init__(self):
        self._index = {}
        self._words = []

    def contains(self, word):
        return word in self._index

    def index(self, word):
        if word in self._index:
            return self._index[word]
        raise ValueError('Word \'%s\' is not in the vocabulary.' % word)

    def word(self, index):
        if index < 0 or index >= len(self._words):
            raise ValueError('Index must be between 0 and %d, found %d instead'
                             % (len(self._words) - 1, index))
        return self._words[index]

    def size(self):
        return len(self._words)

    def items(self):
        return enumerate(self._words)

    def add(self, word):
        if word in self._words:
            return self._index[word]

        index = len(self._words)
        self._words.append(word)
        self._index[word] = index
        return index
