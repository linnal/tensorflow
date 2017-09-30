from dataset_gen import DatasetGen
from dataset_proc import DatasetProc

def main():
  batchSize = 500
  datasetProc = DatasetProc()
  ls = datasetProc.getData()
  print(f'len ls={len(ls)}')
  datasetGen = DatasetGen(ls, batchSize)
  for i in range(0,50):
    d = datasetGen.nextBatch()
    print([len(x) for x,_ in d])

if __name__ == '__main__':
  main()
