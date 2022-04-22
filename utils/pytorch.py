import torch 

def getdevice():
    ''' cuda:0 for gpu vs cpu '''
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tonumpy(tensor):
    ''' forward tensor to cpu, detach, and return as numpy '''
    return tensor.cpu().detach().numpy()