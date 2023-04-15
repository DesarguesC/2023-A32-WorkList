import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



def get_base_argument_parser() -> argparse.ArgumentParser:
    """get the base argument parser for inference scripts"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mini_station_num',
        type=int,
        default=2,
        help='how many tiny stations are on the dataset'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/A32.xlsx',
        help='the path of your dataset'
    )

    parser.add_argument(
        '--pth_path',
        type=str,
        default='./data/weights/3.1scale.pth',
        help='the path of your model weights'
    )
    
    parser.add_argument(
        '--dfg_scale',
        type=float,
        default=3.0,
        help='district-free guidance scale as we dedcribed'
    )

    parser.add_argument(
        '--norm',
        type=str,
        default='standard',
        choices=['standard', 'max-min'],
        help='how to normalize the input data'
    )

    parser.add_argument(
        '--few_shot_mode',
        type=str2bool,
        default=False,
        help='to use zero shot mode'
    )

    parser.add_argument(
        '--few_shot_length',
        type=int,
        default=5,
        help='the length of dataset and must be greater than 5'
    )

    parser.add_argument(
        '--few_shot_path',
        type=str,
        default='./data/few_shot_sample.xlsx',
        help='the path that the few-shot sequence data is placed'
    )

    parser.add_argument(
        '--save_find_fig',
        type=str2bool,
        default=True,
        help='whether to save figure when searching the best scale'
    )

    parser.add_argument(
        '--base_scope',
        type=float,
        default=15.,
        help='use the scope when find the best dfg-scale'
    )

    opt = parser.parse_args()
    return opt




