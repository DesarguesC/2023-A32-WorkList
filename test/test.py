import argparse
from utils.opt_base import get_base_argument_parser as get_parser
from utils.dataset import GetDataset
import torch



def main():
    opt = get_parser()
    M_list, Min_list, Norm_Station_List = GetDataset(opt)
    