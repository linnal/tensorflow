from random import shuffle

class DatasetGen():

  def __init__(self, ls, batchSize):
    self.index = 0
    self.batchSize = batchSize
    self.ls = ls
    self._addExtraData()

  def nextBatch(self):
    i = self.index * self.batchSize

    if i < len(self.ls):
      self.index += 1
      return self.ls[i:i+self.batchSize]

    self.index = 0
    shuffle(self.ls)
    return self.nextBatch()


  def _addExtraData(self):
    dim= self.batchSize - (len(self.ls) % self.batchSize)
    shuffle(self.ls)
    tempLs = self.ls
    shuffle(tempLs)
    self.ls += tempLs[: dim]
