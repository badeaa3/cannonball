import torch
__all__ = ['CoLa', 'LoLa', 'Head', 'CoLaLoLa']

# Combinational Layer
class CoLa(torch.nn.Module):

    def __init__(self, inDim = 10, ncombos = 20, device="cpu"):
        super(CoLa,self).__init__() 
        self.outDim   = inDim + ncombos 
        self.identity = torch.eye(inDim).to(device) 
        self.w_combo  = torch.nn.Parameter(torch.rand(ncombos,inDim))

    def forward(self, x):
        weights = torch.cat([self.identity, torch.nn.Softmax(dim=1)(self.w_combo)], dim=0)
        return torch.einsum('ij, bjk -> bik', weights, x) 


# Loretnz Layer
class LoLa(torch.nn.Module):

    def __init__(self, inDim, device="cpu"):
        super(LoLa, self).__init__()
        
        # list to save objects
        self.eval = False
        self.objs = {key:[] for key in ["m","pt","eta","phi"]}
        self.noutputs = len(self.objs.keys())

    # x shape = [# events] X [# jets] X [# features (E,px,py,pz)]
    def forward(self,x):
        
        m = torch.sqrt(torch.nn.ReLU()(x[:,:,0]**2 - x[:,:,1]**2 - x[:,:,2]**2 - x[:,:,3]**2)) # use relu to remove floating point errors causing m2<0
        pt = torch.sqrt(x[:,:,1]**2 + x[:,:,2]**2)
        eta  = torch.arcsinh(x[:,:,3]/(pt+10**-10)) # add in small epsilon for the zero padded jets
        phi  = torch.atan2(x[:,:,2],x[:,:,1])

        x = torch.stack([m,pt,eta,phi],dim=-1) 
        
        # save predicted objects
        if self.eval:
            self.objs["m"].append(m)
            self.objs["pt"].append(pt)
            self.objs["eta"].append(eta)
            self.objs["phi"].append(phi)

        return x

    def stackobjs(self):
        return {key:torch.cat(val) for key,val in self.objs.items()}

    def clearobjs(self):
        self.objs = {key:[] for key in self.objs}

# The final head of the network
class Head(torch.nn.Module):

    def __init__(self,sizes=None):
        super(Head, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(sizes[i-1], sizes[i]) for i in range(1,len(sizes))])
        self.activations = torch.nn.ModuleList([torch.nn.ReLU() for i in range(len(sizes)-2)] + [torch.nn.Sigmoid()])

    def forward(self,x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x

# Stiches the previous classes together to form CoLa + LoLa + Head 
class CoLaLoLa(torch.nn.Module):            

    def __init__(self, 
                 inDim       = 10,
                 noutputs    = 1,
                 ncombos     = 20,  
                 finalLayers = [200],  
                 device      = "cpu",
                 weights     = None):
        super(CoLaLoLa, self).__init__()

        # Setup network layers
        self.cola        = CoLa(inDim,ncombos,device)
        self.lola        = LoLa(self.cola.outDim,device)
        self.normDim     = self.lola.noutputs*self.cola.outDim 
        self.batchnorm1d = torch.nn.BatchNorm1d(num_features=self.normDim)
        self.head        = Head([self.normDim] + finalLayers + [noutputs])
        
        # send to device
        self.to(device)

        # initialize with weights if present
        if weights: 
            self.load_state_dict(torch.load(weights,map_location=device))      

    def count_parameters(self):
        ''' return the number of parameters inside of model '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        y = self.cola(x)
        y = self.lola(y)
        y = self.batchnorm1d(y.reshape(y.shape[0],-1))
        y = self.head(y)
        # print(y)
        return y

