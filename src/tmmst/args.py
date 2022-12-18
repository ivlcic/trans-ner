import os
from typing import List

import tmmst.const


def dir_path(dir_name) -> str:
    if os.path.isdir(dir_name):
        return dir_name
    else:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            return dir_name
        raise NotADirectoryError(dir_name)


def add_common_dirs(parser) -> None:
    parser.add_argument(
        '-d', '--data_dir', help='Data output directory',
        type=dir_path, default=tmmst.const.default_data_dir
    )
    parser.add_argument(
        '-m', '--models_dir', help='Models directory',
        type=dir_path, default=tmmst.const.default_models_dir
    )


def add_common_arguments(parser) -> None:
    parser.add_argument(
        '-b', '--batch', help='Batch size.', type=int, default=32
    )
    parser.add_argument(
        '--max_seq_len', help='Max sentence length in tokens / words.', type=int, default=256
    )
    parser.add_argument(
        '--no_misc', help='Remove MISC tag (replace i with "O").', action='store_true', default=False
    )
    parser.add_argument(
        '-c', '--limit_cuda_device', help='Limit ops to specific cuda device.', type=int, default=None
    )


def get_pretrained_model_path(args, train: bool = False) -> str:
    if train:
        pt_model_dir = os.path.join(tmmst.const.default_tmp_dir, args.pretrained_model)
        if not os.path.exists(pt_model_dir):
            os.makedirs(pt_model_dir)
        return pt_model_dir
    else:
        return os.path.join(args.models_dir, args.pretrained_model)


def get_tags_to_remove(args) -> List[str]:
    del_misc = []
    if args.no_misc:
        del_misc = ['B-MISC', 'I-MISC']
    return del_misc
