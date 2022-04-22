import torch
import numpy as np
import ROOT 

def invm(X, P, nsig):
    ''' compute the invariant mass given an event labeling. return tensor of masses and tensor of jet labels [m0_c0,m0_c1,m1_c0,mc1_c1,isr...] '''

    # make sure inputs are tensors
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X)
    if not isinstance(P, torch.Tensor):
        P = torch.Tensor(P)

    # send to cpu
    X = X.cpu()
    P = P.cpu()
    
    # extract isr and decay labels
    nchildren   = sum(nsig)
    ndecay      = nchildren - 1
    ndecayones  = nsig[0] - 1
    nnotsig     = X.shape[-1] - nchildren
    isr         = P[:,:-ndecay]
    decay       = P[:,-ndecay:]

    # take the top 4 from the isr 
    indices     = np.argpartition(isr, -4, axis=-1)
    notsig      = indices[:,nnotsig:-nchildren]
    indices     = indices[:,-nchildren:]
    indices,fix = torch.sort(indices) # above sorted max-min but we want to preserve the order
    sig         = indices
    indices     = indices.unsqueeze(2).repeat(1,1,4)
    X           = torch.gather(X,1,indices)

    # set the max value equal to 1 and others to zero
    keep = np.argpartition(decay,-ndecayones, axis=-1)[:,-ndecayones].reshape(-1,1).repeat(1,decay.shape[1])
    mask = torch.Tensor(range(ndecay)).repeat(decay.shape[0],1)
    mask = (mask == keep)
    decay[mask] = 1
    decay[~mask] = 0
    # append one to the front
    decay       = torch.cat((torch.ones(decay.shape[0], 1), decay), 1)

    def uselabels(X, Y, l, nprods=2):
        """ takes in the data tensor and uses the y labels to extract the jets of interest. 
        then return the invariant mass of those jets """
        x = X[Y == l].reshape(X.shape[0], nprods, 4).sum(axis=1)
        return getmass(x)

    # get candidate mass and children
    # NOTE: currently only handles 2 parents
    masses   = []
    children = []
    labels   = [0,1]
    for il in labels:
        masses.append(uselabels(X, decay, il))
        children.append(sig[decay==il].view(P.shape[0],nsig[il]))
    c = torch.cat(children,axis=-1)   # cat children
    c = torch.cat([c,notsig], axis=1) # cat not signal
    m = torch.cat(masses, axis=1)     # cat masses
    
    return m,c


def invmBenchmark(x,y):
    ''' take in jet four momentum and benchmark labels. return the candidate masses. '''
    y_m0 = y[:,[0,1]].long()
    y_m1 = y[:,[2,3]].long()
    x_m0 = x.gather(1,y_m0.unsqueeze(-1).repeat(1,1,4)).sum(axis=1)
    x_m1 = x.gather(1,y_m1.unsqueeze(-1).repeat(1,1,4)).sum(axis=1)
    m0   = getmass(x_m0)
    m1   = getmass(x_m1)
    m    = torch.cat([m0,m1],axis=1)
    return m

def getmass(x):
    ''' three dimension get mass. dim 1 = batch, dim 2 = jets, dim 3 = momentum '''
    return torch.sqrt(
        x[..., :, 0] ** 2
        - x[..., :, 1] ** 2
        - x[..., :, 2] ** 2
        - x[..., :, 3] ** 2).unsqueeze(1)


def drsum(model,x):
    ''' given a MinDeltaRSumHead model get the dr1, dr2, and drsum of truth information '''
    # NOTE: currently only works for unshuffled jets where children are (0,1) and (2,3)
    # if you need to update then pass in the y-information and adjust the etaphi computation
    
    # turn off sortbypt if its on
    temp = model.dr.sortbypt
    model.dr.sortbypt = False

    # get etaphi's of correct children pair
    etaphi0,_,_ = model.dr.fourmom_to_etaphi(x[:,[0,1]])
    etaphi1,_,_ = model.dr.fourmom_to_etaphi(x[:,[2,3]])

    # get delta R's
    ydr0   = model.dr.dR(etaphi0) 
    ydr1   = model.dr.dR(etaphi1)

    # get delta R sum
    ydrsum = model.drs.dRSum(torch.stack([ydr0,ydr1],dim=1))

    # put sort by pt to its previous state
    model.dr.sortbypt = temp

    # return stacked
    return torch.stack([ydr0,ydr1,ydrsum],dim=1)

def DeltaPhi(phi1,phi2):
    ''' delta phi = phi_1 - phi_2 needs to be handled carefully. copying the implementation from root into numpy vectorization 
    https://root.cern.ch/doc/v606/GenVector_2VectorUtil_8h_source.html#l00061 '''
    dphi = phi1 - phi2
    dphi[dphi > np.pi] -= 2.0*np.pi
    dphi[dphi <= -np.pi] += 2.0*np.pi
    return dphi

def BoostToCM(x):
    ''' vectorized using this code as a base https://root.cern.ch/doc/master/GenVector_2LorentzVector_8h_source.html#l00539
    Note: I ignored all of the if statements since I assume that energy and momentum mangitude are nonzero 
    Expected input: batched 4-vectors [[e,px,py,pz]'''
    e = np.expand_dims(x[:, 0],axis=1)
    pxyz = x[: ,1:]
    return -pxyz/e

def Boost(x,b):
    ''' vectorized version of root boost https://root.cern/doc/v608/GenVector_2VectorUtil_8h_source.html#l00329
    Expected input: batched 4-vector [[e,px,py,pz]] '''
    # extract boost vectors
    bx = b[:, 0]
    by = b[:, 1]
    bz = b[:, 2]
    # extract vector to boost
    vt = x[:, 0]
    vx = x[:, 1]
    vy = x[:, 2]
    vz = x[:, 3]

    # check boost
    b2 = bx**2 + by**2 + bz**2
    if np.any(b2 >= 1):
        print("One of the beta vectors is attempting to set Boost speed to >= c")

    # perform boost
    gamma = 1/np.sqrt(1-b2)
    bp = np.multiply(bx,vx) + np.multiply(by,vy) + np.multiply(bz,vz)
    gamma2 = (gamma - 1)/b2
    gamma2[b2<=0] = 0
    x2 = vx + gamma2*bp*bx + gamma*bx*vt
    y2 = vy + gamma2*bp*by + gamma*by*vt
    z2 = vz + gamma2*bp*bz + gamma*bz*vt
    t2 = gamma*(vt+bp)

    # stack and return
    boosted = np.stack([t2,x2,y2,z2],axis=-1)
    return boosted

def boost_to_center_of_mass(u,v):
    ''' boost each pair (u_i,v_i) to their center-of-mass frame 
    Expected input: u and v are batched four vectors [[e,px,py,pz],...] '''
    system = u + v
    boosts = BoostToCM(system)
    u_com  = Boost(u,boosts)
    v_com  = Boost(v,boosts)
    return u_com,v_com,boosts

def boost_to_center_of_mass_root(u,v):
    ''' boost each pair (u_i,v_i) to their center-of-mass frame '''
    ''' implementing slow version with root first '''

    # find the boost vectors to send each system to its center-of-mass frame
    boosts = []
    u_com = []
    v_com = []
    for uu, vv in zip(u,v):
        r_uu = ROOT.Math.PxPyPzEVector(uu[1],uu[2],uu[3],uu[0])
        r_vv = ROOT.Math.PxPyPzEVector(vv[1],vv[2],vv[3],vv[0])
        system = r_uu + r_vv

        # boost 
        boost = system.BoostToCM()
        r_uu_b = ROOT.Math.VectorUtil.boost(r_uu,boost)
        r_vv_b = ROOT.Math.VectorUtil.boost(r_vv,boost)

        # store values
        boosts.append([boost.x(),boost.y(),boost.z()])
        u_com.append([r_uu_b.E(),r_uu_b.Px(),r_uu_b.Py(),r_uu_b.Pz()])
        v_com.append([r_vv_b.E(),r_vv_b.Px(),r_vv_b.Py(),r_vv_b.Pz()])

    return np.array(u_com), np.array(v_com), np.array(boosts)