from dataset_gen import DatasetGen
from dataset_proc import DatasetProc

class DatasetHandler():

  def __init__(self, batchSize, filePath):
    self.datasetProc = DatasetProc(filePath)
    ls = self.datasetProc.getData()
    self.datasetGen = DatasetGen(ls, batchSize)

  def nextTrainBatch(self):
    x, y = self.datasetGen.nextTrainBatch()
    x = self._addExtraEOS(x)
    return x, y

  def nextTestBatch(self):
    x, y = self.datasetGen.nextTestBatch()
    x = self._addExtraEOS(x)
    return x, y

  def hasNextTestBatch(self):
    return self.datasetGen.hasNextTestBatch()

  def vocabSize(self):
    return self.datasetProc.vocabSize()

  def eos(self):
    return self.datasetProc.eos()

  def _addExtraEOS(self, lsbatch):
    max_len = max([len(x) for x in lsbatch])
    return [ls + ([self.datasetProc.eos()] * (max_len - len(ls))) for ls in lsbatch]



if __name__ == '__main__':
  dh = DatasetHandler(4, "../out")
  print(dh.nextTrainBatch())
  print(dh.nextTestBatch())

