import torch
import math
import itertools

__all__ = ['MassAsymmetryHead']

class CoLa(torch.nn.Module):

    def __init__(self,inDim,choose):
        super(CoLa, self).__init__()
        self.inDim    = inDim
        self.choose   = choose
        self.combos   = self.getcombos(inDim,choose)
        self.ncombos  = len(self.combos)
        self.fourvect = 4
        self.inShape  = 3
        
    def getcombos(self,n,k):
        ''' get the possible list indices of n choose k '''
        x = list(set(frozenset(i) for i in itertools.product(range(n), repeat=k) if len(set(i)) == k))
        return torch.tensor(sorted([list(i) for i in x]))

    # return all possible 4-mom pairs
    def forward(self, x):
        self.goodinput(x)
        return torch.stack([torch.sum(x[:,i,:],axis=1) for i in self.combos],dim=1)

    def goodinput(self,x):
        assert len(x.shape)==self.inShape and x.shape[1:]==(self.inDim,self.fourvect)

class LoLa(torch.nn.Module):
    '''returns invariant masses for combined 4-mom
    input:  (N, 6, 4)
    output: (N, 6)'''
    
    def __init__(self,inDim,device):
        super(LoLa, self).__init__()
        self.inDim    = inDim
        self.fourvect = 4
        self.inShape  = 3
        self.metric   = torch.diag(torch.tensor([1., -1., -1., -1.])).to(device)

    # calculate invariant mass
    def forward(self, x):
        self.goodinput(x)
        m2 = torch.einsum('bni,ij,bnj -> bn', x[..., :4] , self.metric, x[..., :4] )
        m  = torch.sqrt(m2)
        return m

    def goodinput(self,x):
        assert len(x.shape)==self.inShape and x.shape[1:]==(self.inDim,self.fourvect)

class MassAsymmetry(torch.nn.Module):

    def __init__(self,inDim,colacombos):
        super(MassAsymmetry, self).__init__()
        self.inDim      = inDim
        self.combos     = self.getcombos(colacombos)
        self.ncombos    = len(self.combos)
        self.inShape    = 2

    def getcombos(self,colacombos):
        ''' get the possible list combinations of cola objects '''
        combos = []
        colacombos = [set(i) for i in colacombos.tolist()]
        for idx,i in enumerate(colacombos):
            for jdx,j in enumerate(colacombos):
                if not i.intersection(j):
                    if [idx,jdx] not in combos and [jdx,idx] not in combos:
                        combos.append([idx,jdx])
        return torch.tensor(sorted(combos))

        # x = list(set(frozenset(i) for i in itertools.product(range(n), repeat=k) if len(set(i)) == k))
        # return torch.tensor(sorted([list(i) for i in x]))

    def forward(self, x):
        self.goodinput(x)
        fwd = torch.stack([self.asym(x[:,i]) for i in self.combos],dim=1)
        # check if there is an increasing number of nan's or infinities at larger values of epsilon
        print(fwd[:,0].shape)
        print(fwd[:,1:].shape)
        print("Fraction of (correct,incorrect,full rows) combinations that are nan: (%1.3f,%1.3f,%1.3f)" % (torch.sum(torch.isnan(fwd[:,0]))/fwd[:,0].numel(),
                                                                                                            torch.sum(torch.isnan(fwd[:,1:]))/fwd[:,1:].numel(),
                                                                                                            torch.sum(torch.sum(~torch.isnan(fwd),dim=1) == 0)/fwd.shape[0])) 
        # the below method of setting nan to inf can be problematic if an entire row is infinite then simple the first element is picked
        # in this case a random guess should be taken since there is no candidate
        j = torch.where(torch.sum(~torch.isnan(fwd),dim=1) == 0)[0] # figure out which rows are all nan
        nNan = torch.sum((torch.sum(~torch.isnan(fwd),dim=1) == 0)) # how many rows are all nan
        idx = torch.randint(0,fwd.shape[1],(1,nNan)) # in each row of all nan's select a random column to set to 1
        fwd[j,idx] = 1

        fwd[torch.isnan(fwd)] = float("inf")
        return fwd

    def asym(self,x):
        return abs(x[:,0] - x[:,1])/(x[:,0]+x[:,1])

    def goodinput(self,x):
        assert len(x.shape)==self.inShape and x.shape[1:]==(self.inDim,)

class SelectionLayer(torch.nn.Module):
    ''' return the index of the minimum value in each row '''
    def __init__(self,inDim):
        super(SelectionLayer, self).__init__()
        self.inDim = inDim # 3
        self.inShape = 2
        
    def forward(self, x):
        self.goodinput(x)
        return torch.argmin(x, dim=1)

    def goodinput(self,x):
        assert len(x.shape)==self.inShape and x.shape[1:]==(self.inDim,)

class MassAsymmetryHead(torch.nn.Module):
    ''' return the index of the minimum value in each row '''
    def __init__(self,inDim,nchildren,device):
        super(MassAsymmetryHead, self).__init__()
        self.inDim     = inDim
        self.nchildren = nchildren
        self.device    = device
        
        # layers
        self.cola = CoLa(inDim,nchildren)
        print(self.cola.combos)
        self.lola = LoLa(self.cola.ncombos,device)
        self.masym = MassAsymmetry(self.cola.ncombos,self.cola.combos)
        print(self.masym.combos)
        self.slay = SelectionLayer(self.masym.ncombos)

    def forward(self, x):
        x = self.cola(x)
        x = self.lola(x)
        m = self.masym(x)
        s = self.slay(m)
        iasym = self.masym.combos[s]
        labs  = torch.flatten(self.cola.combos[iasym,:],start_dim=1).to(self.device)
        ms    = x.gather(1,iasym.to(self.device))
        asym  = m.gather(-1,s.view(-1,1).to(self.device))
        return torch.cat([labs,ms,asym],axis=1)


if __name__ == "__main__":
    # simple test
    jet = [[1.25,0,0,0],[1.5,0,0,0],[0.75,0,0,0],[3,0,0,0],[0.5,0,0,0]]
    jets = torch.tensor([jet]*13)

    inDim     = 5
    nchildren = 2

    mah = MassAsymmetryHead(inDim,nchildren,"cpu")
    out = mah(jets)

