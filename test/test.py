import argparse
import torch
from utils.opt_base import get_base_argument_parser as get_parser
from utils.dataset import GetDataset
from model.model import load_model_NET as load_model
from utils.valid import find_best_scale as find



def main():
    opt = get_parser()
    model = load_model(opt)
    M_list, Min_list, Norm_Station_List = GetDataset(opt)
    if not opt.few_shot_mode:
        M_list, Min_list, Norm_Station_List = GetDataset(opt)
        print('Using model with scale = ', opt.dfg_scale)

    elif opt.few_shot_mode:
        # TODO: find the best scale on few shot
        find(model, opt)
        *_, few_shot_list = GetDataset(opt, few_shot=True)
        

        pass
