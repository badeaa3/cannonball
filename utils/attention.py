# python packages
import torch 
from torch import no_grad
from torch.utils.data import DataLoader
import argparse
import numpy as np
import sys
import os
from datetime import datetime
from progressbar import progressbar
import uproot3
import multiprocessing as mp

# custom code
from hnet.common.utils.pytorch  import getdevice, tonumpy
from hnet.common.utils.data     import dictToRoot
from hnet.common.utils.physics  import invm
from hnet.common.utils.general  import fatal
from hnet.common.logger         import Logger
from hnet.data.batcher          import MyDataset
import hnet.run.train as train
# useful global variables
stime = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

def main():
    ops       = train.options() 
    device    = ops.d if ops.d else getdevice()
    dset      = MyDataset(ops.i,ops.b)
    loader    = DataLoader(dset, num_workers=ops.j)
    model     = train.getmodel(device)
    optimizer = torch.optim.Adam(model.parameters())

    # perform training
    jacobian(**{"model"     :model, 
                "dset"      :dset,
                "loader"    :loader,
                "optimizer" :optimizer,
                "device"    :device})

    

def jacobian(model, dset, loader, optimizer, device):
    ''' perform the full model training '''
    model.eval()
    jacs = []
    preds = []
    i = 0
    for file, data in enumerate(loader):
        trainloader,testloader = dset.prepare(data)
        for X,Y in trainloader:
        	X,Y = X.to(device),Y.to(device)
        	preds.append(model(X))
        	jacs.append(torch.autograd.functional.jacobian(model,X,strict=True))
        	i+=1
        	if i > 10: 
        		break
    	 # for X,Y in testloader:
      #   	X,Y = X.to(device),Y.to(device)
      #   	jacs.append(torch.autograd.functional.jacobian(model,X,strict=True))

    jacs = np.stack(jacs)
    np.savez("jacobians.npz",**{"jacs":jacs,"preds":preds})

# def val(data, model, device, optimizer):
#     ''' validation step given a data (X,Y) and model return the loss value and prediction accuracy '''
#     for X,Y in data:
#         X,Y = X.to(device),Y.to(device)
#         P = model(X)
#         # turn on all gradients
#         X.requires_grad = True
#         # model.eval()
#         jac = torch.autograd.functional.jacobian(model,X,strict=True)
#         print(jac)
#         # attempt another way
#         # optimizer.zero_grad()
#         # for i in range(P.shape[1]):
# 	       #  _loss = P[:,i]
# 	       #  _loss.requires_grad = True
# 	       #  print(_loss)
# 	       #  _loss.backward()
# 	       #  print(_loss.grad)
# 	       #  optimizer.zero_grad()
#         exit()


# def loss(P):
# 	return P

if __name__ == "__main__":
	main()