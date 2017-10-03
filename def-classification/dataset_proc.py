class DatasetProc():

  def __init__(self):
    self.vocab_index = dict()
    self.vocab_word = dict()
    self.maxSentenceLen= 0

  def getData(self):
    ls = []
    with open('out', 'r') as f:
      counter = 0
      maxSentenceLen = 0
      for line in f:
        _id, _source, _content, _def = line.split('\t')
        sentence = _content.split()
        maxSentenceLen = max(maxSentenceLen, len(sentence))
        counter = self._addSentenceToVocab(sentence, counter)

        _def = int(_def.strip())
        ls.append((self._translateWords(sentence), _def))

    self.maxSentenceLen= maxSentenceLen
    return [(s+ self._extraEos(maxSentenceLen - len(s)),d) for (s,d) in ls ]

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

  def _extraEos(self, dim):
    return [self.eos()] * dim

