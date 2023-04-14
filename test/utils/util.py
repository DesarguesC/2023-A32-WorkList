import numpy as np
import torch


def normalization(x: list):
    M, m = np.max(x), np.min(x)
    for i in range(len(x)):
        x[i] = (x[i] - (M + m) / 2) / ((M - m) / 2)
    # x in [-1, 1]
    return M, m, x

def cal(x: list):
    mu, sigma = 0, 0
    for i in range(len(x)):
        mu += x[i]
    mu /= len(x)*1.0
    for i in range(len(x)):
        sigma += (x[i]-mu) ** 2
    sigma /= len(x)*1.0
    if sigma==0:
        print(x)
    from math import sqrt
    for i in range(len(x)):
        x[i] = (x[i]-mu) / sqrt(sigma)

    if sigma==0:
        print(x)

    return mu, sigma, x

def ArrNorm(x: np.ndarray, state='max-min'):
    assert isinstance(x, np.ndarray), "We need a np.ndarray"
    
    if state == 'max-min':
        M_list, m_list, res = [], [], []
        for i in range(x.shape[0]):
            u = x[i].tolist()
            M, m, t = normalization(u)
            res.append(t)
            M_list.append(M)
            m_list.append(m)
        return M_list, m_list, np.array(res)
    elif state == 'standard':
        mu_list, sigma_list, res = [], [], []
        for i in range(x.shape[0]):
            u = x[i].tolist()
            m, s, t = cal(u)
            res.append(t)
            mu_list.append(m)
            sigma_list.append(s)
        return mu_list, sigma_list, res


def df2arr(x) -> np.ndarray:
    return np.array(x, dtype=np.float32)





