import argparse
import torch
from utils.opt_base import get_base_argument_parser as get_parser
from utils.dataset import GetDataset, load_array
from utils.valid import find_best_scale, test_model
from model.model import load_model_NET as load_model
from utils.valid import find_best_scale as find




def main():
    opt = get_parser()
    model = load_model(opt)
    M_list, Min_list, Station_Data, Norm_Station_List = GetDataset(opt)
    L = len(Station_Data)
    Test_Data_Iter = load_array((Station_Data, Norm_Station_List[0:L-1]), batch_size=1)
    # Station_Data: input
    # Norm_Station_List: output
    if not opt.few_shot_mode:
        print('Using model with scale = ', opt.dfg_scale)

    elif opt.few_shot_mode:
        # TODO: find the best scale on few shot
        find(model, opt)
        *_, few_shot_input, few_shot_output = GetDataset(opt, few_shot=True)
        few_shot_data_iter = load_array((few_shot_input, few_shot_output), batch_size=len(few_shot_input))
        R_list, scale_list = find(model, opt, few_shot_data_iter)
        R_max = max(R_list)
        scale = scale_list[R_list.index(R_max)]
        print('Using model with scale = {0} found on few-shot data'.format(scale))
        model.reset(scale)
        
    valid_R, use_scale = test_model(model, Test_Data_Iter)
    print('With using district-free scale, we found following R-squared during validatioin.')
    print(valid_R)


    
if __name__ == "__main__":
    main()

