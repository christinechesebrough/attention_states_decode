import numpy as np

def recurse_np2list(obj):
    """Convert a numpy array with nested numpy array and dictionaries to a list"""

    if isinstance(obj,dict):
        for k in obj.keys():
            obj[k] = recurse_np2list(obj[k])
        return obj
    elif isinstance(obj,np.ndarray):
        obj = obj.tolist()
        obj = recurse_np2list(obj)
        return obj
    elif isinstance(obj,list):
        l = len(obj)
        if l > 0:
            for ii in range(l):
                obj[ii] = recurse_np2list(obj[ii])
        return obj
    elif isinstance(obj,tuple):
        obj = list(obj)
        obj = recurse_np2list(obj)
        return obj
    elif obj is None:
        print('Help! NoneType')
        print(obj)
    else:
        return obj
