from dataset_gen import DatasetGen
from dataset_proc import DatasetProc

def main():
  batchSize = 50
  datasetProc = DatasetProc()
  ls = datasetProc.getData()
  print(f'len of dataset={len(ls)}')
  datasetGen = DatasetGen(ls, batchSize)
  for i in range(0,50):
    d = datasetGen.nextTrainBatch()
    print([len(x) for x,_ in d])

if __name__ == '__main__':
  main()
