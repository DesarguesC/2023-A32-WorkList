import math
from scipy import stats



def rsquared(x, y): 
    length = x.shape[-1]
    assert x.shape==y.shape, "Unequal Shape Error"
    r, x, y = [], x.detach(), y.detach()
    x, y = x.mean(dim=[0,1], keepdim=False), y.mean(dim=[0,1], keepdim=False)
    for i in range(length):
        _, _, r_value, _, _ = stats.linregress(x[:,i].detach().numpy(), y[:,i].detach().numpy()) 
        r.append(r_value ** 2)
    return r




