import argparse
import torch
from utils.opt_base import get_base_argument_parser as get_parser
from utils.dataset import GetDataset, load_array
from utils.valid import find_best_scale, test_model
from model.model import load_model_NET as load_model
from utils.valid import find_best_scale as find





def main():
    opt = get_parser()
    model_list = load_model(opt)
    assert len(model_list) == 5
    M_list, Min_list, I_O_Menu, Standard_Data = GetDataset(opt)
    
    Test_Data_Iter_List = [load_array(I_O_Menu[i], batch_size=len(I_O_Menu[i][0])) for i in range(len(I_O_Menu))]
    print(I_O_Menu[0][0].shape)
    # Station_Data: input
    # Norm_Station_List: output
    if not opt.few_shot_mode:
        print('Using model with scale = ', opt.dfg_scale)

    elif opt.few_shot_mode:
        # TODO: find the best scale on few shot
        RR, SS = [], []
        # find(model, opt)
        *_, few_shot_input, few_shot_output = GetDataset(opt, few_shot=True)
        few_shot_data_iter = load_array((few_shot_input, few_shot_output), batch_size=len(few_shot_input))
        for i in range(len(model_list)):
            model = model_list[i]    
            R_list, scale_list = find(model, opt, few_shot_data_iter)
            R_max = max(R_list)
            scale = scale_list[R_list.index(R_max)]
            RR.append(R_list)
            SS.append(scale_list)
            print('Using model with scale = {0} for feature-{1}'.format(scale, i+1))
            model_list[i].reset(scale)
    
    print('Start Validation....')
    for i in range(len(Test_Data_Iter_List)):
        valid_R, use_scale = test_model(model_list, Test_Data_Iter_List[i])
        print('With using district-free scale = %.5f, we found R-squared in DISTRICT %d'%(use_scale, i+1))
        print(valid_R)


    
if __name__ == "__main__":
    main()

