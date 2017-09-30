class DatasetProc():

  def __init__(self):
    self.vocab_wordid = dict()
    self.vocab_idword = dict()

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
    test = [(s+ self._extraEos(maxSentenceLen - len(s)),d) for (s,d) in ls ]
    return [(s+ self._extraEos(maxSentenceLen - len(s)),d) for (s,d) in ls ]

  def _addSentenceToVocab(self, sentence, counter):
    for word in sentence:
      if word not in self.vocab_wordid:
        self.vocab_wordid[word] = counter
        self.vocab_idword[counter] = word
        counter += 1
    return counter

  def _translateWords(self, sentence):
    return [self.vocab_wordid[word] for word in sentence ]

  def _extraEos(self, dim):
    return [self._eos()] * dim

  def _eos(self):
    return self.vocab_wordid['.']
