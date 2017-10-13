class DatasetProc():

  def __init__(self, filePath):
    self.vocab_index = dict()
    self.vocab_word = dict()
    self.filePath = filePath

  def getData(self):
    ls = []
    with open(self.filePath, 'r') as f:
      counter = 0
      for line in f:
        _id, _source, _content, _def = line.split('\t')
        sentence = _content.split()
        counter = self._addSentenceToVocab(sentence, counter)

        _def = int(_def.strip())
        ls.append((self._translateWords(sentence), _def))

    return ls

  def vocabSize(self):
    return len(self.vocab_index.keys())

  def eos(self):
    return self.vocab_index['.']

  def _addSentenceToVocab(self, sentence, counter):
    for word in sentence:
      if word not in self.vocab_index:
        self.vocab_index[word] = counter
        self.vocab_word[counter] = word
        counter += 1
    return counter

  def _translateWords(self, sentence):
    return [self.vocab_index[word] for word in sentence ]


