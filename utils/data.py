import uproot3 
import numpy as np
from hnet.common.utils.general import fatal

def dictToRoot(oname, data):
    ''' Convert a dictionary to a root file using uproot3. Currently supported dictionary formats:
    1) dictionary of dictionaries 
    ex. data = {"tree1": {"branch1": ..., "branch2":...},
                "tree2": ...}
    2) dictionary of lists or numpy arrays
    ex. data = {"branch1": np.array([...]),
                "branch2": ...,
                "branch3": ....} 
    '''

    # separate input dictionary into list of trees, branches, data
    tnames   = []
    bnames   = []
    bdata    = []
    for key,val in data.items():
        if isinstance(val,dict):
            tnames.append(key)
            bnames.append({k:"float64" for k,v in val.items()})
            bdata.append({k:np.array(v) for k,v in val.items()}) 
        elif isinstance(val,np.ndarray) or isinstance(val,list):
            bnames.append({key:"float64"})
            bdata.append(np.array(val)) 

    # check that branches are the same length
    for t,b in zip(tnames,bdata):
        lengths = np.array([val.shape for key,val in b.items()])
        if not np.all(lengths == lengths[0]):
            print(b.keys())
            fatal("Tree %s -- not all branches are the same length: %s" % (t,lengths))

    # if no tree name was provided
    if len(tnames) == 0: 
        tnames.append("tree")

    # create and write to root file
    with uproot3.recreate(oname) as f:
        print("Writing %s" % oname)
        for t,b,d in zip(tnames,bnames,bdata):
            print("working on ttree %s" % (t))
            f[t] = uproot3.newtree(b)
            f[t].extend(d)

    return oname 


