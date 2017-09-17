from random import shuffle

class DatasetProcessor():

  def __init__(self):
    lsSentences = self.listOfSentences();
    self.vocab_word, self.vocab_tag, self.vocab_word_id, self.vocab_tag_id = self.createVocabolary(lsSentences)
    lsSentences = self.applyVocab(lsSentences)

    max_len = self.findMaxLen(lsSentences)
    self.balanceLsSentences = [self.addEndOfSequenze(max_len, sentece) for sentece in  lsSentences]

  def applyVocab(self, ls):
    print('applyVocab')
    outputls = []
    for sentence in ls:
      trans_s = []
      for wordtag in sentence:
        w = wordtag.split('/')
        if len(w) > 2 or len(w) <= 1:
          continue
        word, tag = w
        trans_wt = f'{self.vocab_word[word]}/{self.vocab_tag[tag]}'
        trans_s.append(trans_wt)
      outputls.append(trans_s)
    return outputls

  def createVocabolary(self, ls):
    print("createVocabolary")
    vocab_w = dict()
    vocab_t = dict()
    vocab_w_id = dict()
    vocab_t_id = dict()

    count_w = 0
    count_t = 0
    for sentence in ls:
      for wordtag in sentence:
        w = wordtag.split('/')
        if len(w) > 2 or len(w) <= 1:
          continue
        word, tag = w
        if word not in vocab_w:
          vocab_w[word] = count_w
          vocab_w_id[count_w] = word
          count_w += 1
        if tag not in vocab_t:
          vocab_t[tag] = count_t
          vocab_t_id[count_t] = tag
          count_t += 1

    vocab_w['.'] = count_w
    vocab_w_id[count_w] = '.'
    vocab_t['EOS'] = count_t
    vocab_t_id[count_t] = 'EOS'

    return vocab_w, vocab_t, vocab_w_id, vocab_t_id

  def listOfSentences(self):
    print('listOfSentences')
    output_word_tag_list= []

    with open("corpus.small") as f:
      lines = [p for p in f]
      txt = ' '.join(lines)
      sentence = txt.split('./.')
      for s in sentence:
        ls_line = s.split()
        if(len(ls_line) == 0):
          continue
        output_word_tag_list.append(ls_line)
    return output_word_tag_list


  def findMaxLen(self, data):
    print('findMaxLen')
    lens = [len(x) for x in data]
    return max(lens)

  def addEndOfSequenze(self, max_len, ls):
    diff = max_len - len(ls)
    # if diff == 0:
    #   print ([ ( self.vocab_word_id[int(x[0])], self.vocab_tag_id[int(x[1])]) for x in [x.split("/") for x in ls] ])
    ls += [f'{self.vocab_word["."]}/{self.vocab_tag["EOS"]}']*diff
    return ls

  def getTestTrainDataset(self, ls):
    print('getTestTrainDataset')
    shuffle(ls)
    index = int( len(ls)*0.8 )
    train_data = ls[0:index]
    test_data = ls[index:]
    return train_data, test_data

  def separateWordsFromTags(self, ls):
    print('separateWordsFromTags')
    output_words_list = []
    output_tags_list = []

    for sentece_ls in ls:
      words_list = []
      tags_list = []
      for word_tag in sentece_ls:
        w = word_tag.split('/')
        if len(w) > 2 or len(w) <= 1:
          continue
        word, tag = w
        words_list.append(int(word))
        tags_list.append(int(tag))
      output_words_list.append(words_list)
      output_tags_list.append(tags_list)

    return output_words_list, output_tags_list

  def perpareDataset(self):
    print('perpareDataset')
    train, test = self.getTestTrainDataset(self.balanceLsSentences)

    train_words, train_tags = self.separateWordsFromTags(train)
    test_words, test_tags = self.separateWordsFromTags(test)

    return train_words, train_tags, test_words, test_tags


  def getExtraTrainData(self, dim):
    shuffle(self.balanceLsSentences)
    extra = self.balanceLsSentences[:dim]
    return self.separateWordsFromTags(extra)

  def getExtraTestData(self, dim):
    shuffle(self.balanceLsSentences)
    extra = self.balanceLsSentences[:dim]
    return self.separateWordsFromTags(extra)


  def translate(self, ls, _type="tag"):
    if _type == 'word':
      return [self.vocab_word_id[x] for x in ls]

    return [self.vocab_tag_id[x] for x in ls]

