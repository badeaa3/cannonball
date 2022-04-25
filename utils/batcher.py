'''
batch manager for handling a list of files on input. loads them asynchronously to be ready when called on. 
'''

from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch
import os
import glob
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from time import time

class MyDataset(Dataset):
    def __init__(self,data,batch_size,background=False):
        self.data_files = self.input(data)
        self.batch_size = batch_size
        self.background = background

    def __getitem__(self, idx):
        return np.load(self.data_files[idx],allow_pickle=True)

    def __len__(self):
        return len(self.data_files)

    def input(self,data):
        if os.path.isfile(data) and ".npz" in os.path.basename(data):
            return [data]
        elif os.path.isfile(data) and ".txt" in os.path.basename(data):
            return sorted([line.strip() for line in open(data,"r")])
        elif os.path.isdir(data):
            return sorted(os.listdir(data))
        elif "*" in data:
            return sorted(glob.glob(data))
        return []

    def prepare(self,data):
        jets   = data["jets"][0].float() # if converting to float here slows things down eventually need to fix 
        labels = data["labels"][0].float() if not self.background else torch.zeros(jets.shape[0],1)
        x_train, x_test, y_train, y_test = train_test_split(jets, labels, shuffle=False)
        trainset = list(zip(x_train, y_train))
        testset  = list(zip(x_test , y_test ))
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        testloader  = DataLoader(testset , batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        return trainloader, testloader


class MyDataLoader():
    def __init__(self, data, nworkers, batch_size):
        self.dset = MyDataset(data)
        self.loader = DataLoader(dset, num_workers=nworkers)

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", help="File to batch over.", default=".", required=True)
    parser.add_argument("-b", help="Batch size", default=1, type=int)
    parser.add_argument("-j", help="Number of works used by DataLoader. This could significantly affect run time. ", default=0, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    ops    = options()
    dset   = MyDataset(ops.i,ops.b)
    loader = DataLoader(dset, num_workers=ops.j)
    s      = time()
    tot    = 0

    for file, data in enumerate(loader):
        start = time()
        trainloader,testloader = dset.prepare(data)
        for d in trainloader: 
            continue
        for d in testloader: 
            continue
        tmp = time()-start
        tot += tmp
        print(f"File {file} took {tmp:1.6f} seconds")

    print("The loop took: %1.6f seconds"%(tot))
    print("The entire run took: %1.6f seconds"%(time()-s))
