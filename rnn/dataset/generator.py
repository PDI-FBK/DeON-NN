from random import shuffle


class Generator():

    def __init__(self, ls, batchSize):
        self.batchSize = batchSize
        self.ls = ls

        self.initTrainTest()
        print('{className}: train= {trainLen}, test={testLen}'
            .format(className= type(self).__name__, trainLen= len(self.train), testLen= len(self.test)))

    def initTrainTest(self):
        print('{className}: shuffle and initTrainTest'
            .format(className= type(self).__name__))
        self.trainIndex = 0
        self.testIndex = 0
        self.train, self.test = self._splitDataset() # 80%,20%
        self.train = self._addExtraData(self.train)
        self.test = self._addExtraData(self.test)

    def nextTrainBatch(self):
        i = self.trainIndex * self.batchSize

        if i < len(self.train):
            self.trainIndex += 1
            x,y = self._getBatch(self.train, i, i+self.batchSize)
            return x,y

        self.initTrainTest()
        return self.nextTestBatch()

    def nextTestBatch(self):
        i = self.testIndex * self.batchSize
        if i < len(self.test):
            self.testIndex += 1
            x,y = self._getBatch(self.test, i, i+self.batchSize)
            return x,y

        self.testIndex = 0
        return self.nextTestBatch()

    def hasNextTestBatch(self):
        i = self.testIndex * self.batchSize
        if i < len(self.test):
            return True

        self.testIndex = 0
        return False

    def _getBatch(self, ls, _from, _to):
        lsbatch = ls[_from:_to]
        x = [s for s, _ in lsbatch]
        y = [d for _, d in lsbatch]
        return x, y

    def _splitDataset(self):
        shuffle(self.ls)
        defs = []
        nodefs = []
        for sentence, _def in self.ls:
            if _def == 1:
                defs.append((sentence, _def))
            else:
                nodefs.append((sentence, _def))

        dim_train_def, dim_train_nodef = int(len(defs)*0.80), int(len(nodefs)*0.80)

        train = defs[:dim_train_def] + nodefs[:dim_train_nodef]
        test = defs[dim_train_def:] + nodefs[dim_train_nodef:]

        return train, test

    def _addExtraData(self, ls):
        dim = self.batchSize - (len(ls) % self.batchSize)
        tempLs = ls
        shuffle(tempLs)
        ls += tempLs[: dim]
        return ls
