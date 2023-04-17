import math
import numpy as np
from scipy import stats
import torch
from torch.nn import Module
from tqdm import tqdm
from model.model import NET


# def rsquared(x, y): 
#     length = x.shape[-1]
#     assert x.shape==y.shape, "Unequal Shape Error"
#     r, x, y = [], x.detach(), y.detach()
#     x, y = x.mean(dim=[0], keepdim=False), y.mean(dim=[0], keepdim=False)
#     print('x.shape = ', x.shape)
#     for i in range(length):
#         _, _, r_value, _, _ = stats.linregress(x[i].detach().numpy().tolist(), y.detach().numpy()) 
#         r.append(r_value ** 2)
#     return r

def rsquared(x, y): 
    _, _, r_value, _, _ = stats.linregress(x.detach().cpu().numpy(), y.detach().cpu().numpy()) 
    return r_value**2


def find_best_scale(model: Module, opt, data_iter):
    # TODO: find best scale for the best r-squared.

    ###########################################################
    #                                                         #
    # we are now in few-shot mode                             #
    # model: loaded class NET                                 #
    # opt: inputs / condition                                 #
    # data_iter: data iterator loading few-shot datasets      #
    #                                                         #
    ############################################################

    print('Start to find scale for each feature index')

    R_list = []
    R_plt_list = []
    scale_list = []
    scale_plt_list = []
    opt.base_scope = -opt.base_scope if opt.base_scope<0 else opt.base_scope
    for idx in range(5):
        print('Finding: [%d|%d]' % (idx, 5))
        r = .0
        r_plt = []
        scale = .0
        scale_plt= []
        for scale in np.arange(-opt.base_scope, opt.base_scope, .1):
            with torch.no_grad():
                for i, use in enumerate(data_iter):
                    pred = model(use[0].cuda() if torch.cuda.is_available() else use[0])
                    try:
                        pred = pred.reshape(-1, 1, 5)
                    except:
                        raise RuntimeError('Invalid sequence length or batch size')
                    use[1] = use[1].reshape(-1, 1, 5)
                    assert pred.shape == use[1].shape, 'unequal shape error'
                    r1 = rsquared(pred[idx], use[1][idx])
            r_plt.append(r)
            scale_plt.append(scale)
            if r1 > r:
                r = r1 
                scale1 = scale
        R_list.append(r)
        scale_list.append(scale1)
        R_plt_list.append(r_plt)
        scale_plt_list.append(scale_plt)

    assert len(R_list)==len(scale_list)==len(R_plt_list)==len(scale_plt_list)==5, \
                'Invalid valuation with for length: {0}, {1}, {2}, {3}'.\
                    format(len(R_list), len(scale_list), len(R_plt_list), len(scale_plt_list))

    if opt.save_finfd_fig:
        import matplotlib.pyplot as plt
        for i in range(5):
            plt.figure(i)
            plt.scatter(scale_plt_list, R_plt_list[i], s=5, color='blue', marker='*')
            plt.plot(scale_plt_list, R_plt_list, 'r-.')
            plt.xlabel('dfg-scale')
            plt.ylabel('R-squared')
            plt.title('Find the best R-squared through scale')
            plt.legend()
            plt.savefig('./FIND/idx={0}.png'.format(i))
    print('finish searching process on few-shot dataset with following value: ')
    print(R_list, scale_list)
    return R_list, scale_list

def test_model(model_list: list, data_iter):
    R_list, scale_list = [], []
    assert len(model_list) == 5, 'model_list length Error'
    with torch.no_grad():
        for idx in range(len(model_list)):
            model = model_list[idx]
            scale_list.append(model.scale)
            for i, use in enumerate(tqdm(data_iter)):
                # print('\nuse[0].shape = {0}, use[1].shape = {1}'.format(use[0].shape, use[1].shape))
                assert i==0, 'batch size of data iterator wrong set!'
                assert isinstance(model, NET), 'Class not match Error'
                pred = model(use[0].cuda() if torch.cuda.is_available() else use[0])
                try:
                    pred = pred.reshape(-1, 5)
                    use[1] = use[1].reshape(-1, 5)
                except:
                    raise RuntimeError('Feature amount of the input must be 5')
                assert pred.shape == use[1].shape, 'unequal shape error'
                # print(pred.shape)
                R = rsquared(pred[idx], use[1][idx])
                R_list.append(R)
    assert len(R_list) == 5, 'Err'
    return R_list, scale_list
