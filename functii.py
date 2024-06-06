import numpy as np

def calcul_criterii(alpha, procent_minimal=70):
    m=len(alpha)
    procent_cumulat=np.cumsum(alpha)*100/m;
    k1=np.where(procent_cumulat>procent_minimal)[0][0]+1
    k2=len(np.where(alpha>1)[0])
    eps=alpha[:m-1]-alpha[1:]
    sigma=eps[:m-2]-eps[1:]
    exista_negative=sigma<0
    if any(exista_negative):
        k3=np.where(exista_negative)[0][0]+2
    else:
        k3=np.NAN
    return(k1,k2,k3)