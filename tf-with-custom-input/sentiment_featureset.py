import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import random
import pickle #save the processed data


def main():
  featureset = FeatureSet(['pos.txt', 'neg.txt'], 10000000)
  train_x, train_y, test_x, test_y = featureset.create_feature_sets_and_labels()
  with open('sentiment_set.pickle', 'wb') as f:
    pickle.dump([train_x, train_y, test_x, test_y], f)


class FeatureSet(object):
  def __init__(self, files, lines):
    self.files = files
    self.lemmatizer = WordNetLemmatizer()
    self.lines = lines

  def create_feature_sets_and_labels(self, test_size=0.1):
    features = []
    for i, f in enumerate(self.files):
      lexicon = self.create_lexicon(f)
      classification = [0] * len(self.files)
      classification[i] = 1
      features += self.sample_handling(f, lexicon, classification)

    random.shuffle(features)

    features = np.array(features)
    testing_size = int(test_size * len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x  = list(features[:,0][-testing_size:])
    test_y  = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y

  def sample_handling(self, filename, lexicon, classification):
    featureset = []
    for tokens in self.get_tokens(filename):
      lexicon = [self.lemmatizer.lemmatize(i) for i in tokens]
      features = np.zeros(len(lexicon))
      for word in tokens:
        if word in lexicon:
          index = lexicon.index(word)
          features[index] += 1
      features = list(features)
      featureset.append([features, classification])

    return featureset

  def create_lexicon(self, filename):
    result = []
    for tokens in self.get_tokens(filename):
      words = [word for word in tokens if word not in stopwords.words('english')]
      lexicon = [self.lemmatizer.lemmatize(i) for i in words]
      result += lexicon
    return result


  def get_tokens(self, filename):
    with open(filename, 'r') as f:
      contents = f.readlines()
      for line in contents[:self.lines]:
        words = list(word_tokenize(line.lower()))
        yield words


if __name__ == '__main__':
  main()
