from random import shuffle

class DatasetProcessor():

  def __init__(self):
    lsSentences = self.listOfSentences();
    self.vocab = self.createVocabolary(lsSentences)
    lsSentences = self.applyVocab(lsSentences, self.vocab)

    max_len = self.findMaxLen(lsSentences)
    self.balanceLsSentences = [self.addEndOfSequenze(max_len, sentece) for sentece in  lsSentences]

  def applyVocab(self, ls, vocab):
    print('applyVocab')
    outputls = []
    for sentence in ls:
      trans_s = []
      for wordtag in sentence:
        w = wordtag.split('/')
        if len(w) > 2 or len(w) <= 1:
          continue
        word, tag = w
        trans_wt = f'{vocab[word]}/{vocab[tag]}'
        trans_s.append(trans_wt)
      outputls.append(trans_s)
    return outputls

  def createVocabolary(self, ls):
    print("createVocabolary")
    vocab = dict()
    count = 0
    for sentence in ls:
      for wordtag in sentence:
        w = wordtag.split('/')
        if len(w) > 2 or len(w) <= 1:
          continue
        word, tag = w
        if word not in vocab:
          vocab[word] = count
          count += 1
        if tag not in vocab:
          vocab[tag] = count
          count += 1

    vocab['.'] = count + 1
    vocab['EOS'] = count + 2
    return vocab

  def listOfSentences(self):
    print('listOfSentences')
    output_word_tag_list= []

    with open("corpus.small") as f:
      for paragraph in f:
        paragraph = paragraph.split("./.")
        for line in paragraph:
          ls_line = line.split()
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
    ls += [f'{self.vocab["."]}/{self.vocab["EOS"]}']*diff
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


