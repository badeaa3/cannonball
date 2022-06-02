import torch
import math
import itertools
import sys

# custom code
sys.path.insert(0,"utils")
sys.path.insert(0,"../utils")
import physics as phys

class DeltaR(torch.nn.Module):

    def __init__(self,inDim,choose,sortbypt=False,topk=None):
        super(DeltaR, self).__init__()
        self.inDim    = inDim
        self.choose   = choose
        self.combos   = self.getcombos(topk if topk else inDim,
                                       choose)
        self.ncombos  = len(self.combos)
        self.fourvect = 4
        self.inShape  = 3
        self.sortbypt = sortbypt
        self.topk     = topk
        self.ind      = None
        self.passEtaCut  = None
        self.passPtCut   = None

    def forward(self,x):
        ''' take in batched events of (e,x,y,z) and return a list of dR between all jet pairings per event '''
        self.goodinput(x)
        etaphi, pt, self.ind = self.fourmom_to_etaphi(x)
        self.passEtaCut = (torch.sum(abs(etaphi[:,:,0]) < 2.4,dim=1) >= 4)
        self.passPtCut  = torch.flatten((torch.sum(pt > 120,dim=1) >= 4))
        fwd = torch.stack([self.dR(etaphi[:,i,:]) for i in self.combos],dim=1)
        return fwd

    def getcombos(self,n,k):
        ''' get the possible list indices of n choose k '''
        x = list(set(frozenset(i) for i in itertools.product(range(n), repeat=k) if len(set(i)) == k))
        return torch.tensor(sorted([list(i) for i in x]))

    def goodinput(self,x):
        ''' ensure that the input is the expected size '''
        assert len(x.shape)==self.inShape and x.shape[1:]==(self.inDim,self.fourvect)

    def fourmom_to_etaphi(self,x):
        ''' take in batched events of 4-momentum (e,px,py,pz) and return (eta,phi) per jet per event (possibly ordered by pt), pt, and pt ordering indices '''
        px     = x[:,:,1]
        py     = x[:,:,2]
        pz     = x[:,:,3]
        pt     = torch.sqrt(px**2 + py**2)
        eta    = torch.arcsinh(pz/pt)
        phi    = torch.atan2(py,px)
        etaphi = torch.stack([eta,phi],dim=-1)
        ind    = None
        if self.sortbypt:
            etaphi, pt, ind = self.ptsort(etaphi,pt,self.topk)
        return etaphi, pt, ind

    def dR(self,x):
        ''' take in batched (eta,phi) for two jets and return their delta R''' 
        eta  = x[:,:,0]
        deta = eta[:,0] - eta[:,1]
        phi  = x[:,:,1]
        dphi = phys.DeltaPhi(phi[:,0],phi[:,1])
        return torch.sqrt(deta**2 + dphi**2)

    def ptsort(self,x,pt,topk):
        ''' take in batched four momentum x and pt of that data. Sort x by pt and return the topk leading jets '''
        ptsorted, ind = torch.topk(pt,topk)
        nobs          = x.shape[-1]
        ind           = ind.unsqueeze(-1)
        ptsorted      = ptsorted.unsqueeze(-1)
        return x.gather(1,ind.repeat_interleave(nobs,-1)), ptsorted, ind

class DeltaRSum(torch.nn.Module):

    def __init__(self,inDim,drcombos,const):
        super(DeltaRSum, self).__init__()
        self.inDim   = inDim
        self.combos  = self.getcombos(drcombos)
        self.ncombos = len(self.combos)
        self.inShape = 2
        self.const   = const

    def forward(self,x):
        ''' take in batched lists of delta R and return the minimum sum from all combinations'''
        self.goodinput(x)
        fwd = torch.stack([self.dRSum(x[:,i]) for i in self.combos],dim=-1)
        fwd[torch.isnan(fwd)] = float("inf")
        return fwd

    def getcombos(self,drcombos):
        ''' get the possible list indices of combinations '''
        combos = []
        drcombos = [set(i) for i in drcombos.tolist()]
        for idx,i in enumerate(drcombos):
            for jdx,j in enumerate(drcombos):
                if not i.intersection(j):
                    if [idx,jdx] not in combos and [jdx,idx] not in combos:
                        combos.append([idx,jdx])
        return torch.tensor(sorted(combos))

    def goodinput(self,x):
        ''' ensure that the input is the expected size '''
        assert len(x.shape)==self.inShape and x.shape[1:]==(self.inDim,)

    def dRSum(self,x):
        ''' return the sum (dR-c) for two dR '''
        return torch.sum(torch.abs(x-self.const),axis=1)

class SelectionLayer(torch.nn.Module):
    ''' return the index of the minimum value in each row '''
    def __init__(self,inDim):
        super(SelectionLayer, self).__init__()
        self.inDim = inDim # 3
        self.inShape = 2
        
    def forward(self, x):
        self.goodinput(x)
        fwd = torch.argmin(x, dim=1)
        return fwd

    def goodinput(self,x):
        assert len(x.shape)==self.inShape and x.shape[1:]==(self.inDim,)

class MinDeltaRSumHead(torch.nn.Module):

    def __init__(self,
                 inDim,
                 nchildren,
                 const,
                 device,
                 sortbypt=False,
                 topk=None):
        super(MinDeltaRSumHead, self).__init__()
        self.inDim     = inDim
        self.nchildren = nchildren
        self.const     = const
        self.device    = device

        # layers
        self.dr   = DeltaR(inDim,nchildren,sortbypt,topk)
        self.drs  = DeltaRSum(self.dr.ncombos,self.dr.combos,const)
        self.slay = SelectionLayer(self.drs.ncombos)

    def forward(self,x):
        ''' take in batched lists of delta R and return the minimum sum from all combinations'''
        x = self.dr(x)
        m = self.drs(x)
        s = self.slay(m)
        imdrs = self.drs.combos[s]
        labs  = torch.flatten(self.dr.combos[imdrs,:],start_dim=1).to(self.device)
        if self.dr.sortbypt:
            labs = self.smart_sort(torch.flatten(self.dr.ind,start_dim=1),labs)
        dr    = x.gather(1,imdrs.to(self.device))
        mdrs  = m.gather(-1,s.view(-1,1).to(self.device))
        fwd   = torch.cat([labs,dr,mdrs],axis=1)
        return fwd

    def smart_sort(self, x, permutation):
        ''' sort tensor x using the indices in tensor permutaitons.
        taken from: https://discuss.pytorch.org/t/how-to-sort-tensor-by-given-order/61625/2 '''
        d1, d2 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
            permutation.flatten()
        ].view(d1, d2)
        return ret

if __name__ == "__main__":
    # simple test
    jet = [[1.25,1,1,1],[0.5,1,0,0],[1.5,1,1,1],[0.75,2,2,2],[3,2,2,2]]
    jets = torch.tensor([jet]*2)

    inDim     = 5
    nchildren = 2
    const     = 0
    device    = "cpu"

    mdrsh = MinDeltaRSumHead(inDim,nchildren,const,device)
    out = mdrsh(jets)




