import os
import logging
from typing import List, Dict, Any

import tmmst.const

logger = logging.getLogger('args')
logger.addFilter(tmmst.fmt_filter)


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
        '--pro', help='Enable Product (PRO) tag.', action='store_true', default=False
    )
    parser.add_argument(
        '--evt', help='Enable Event (EVT) tag.', action='store_true', default=False
    )
    parser.add_argument(
        '-c', '--limit_cuda_device', help='Limit ops to specific cuda device.', type=int, default=None
    )


def chech_param(conf: Dict, p_name: str) -> Any:
    p = conf.get(p_name)
    if not p:
        logger.warning('Missing [%s] param in [%s] config', p_name, conf)
        exit(1)
    return p


def chech_dir_param(conf: Dict, param_name: str, parent_path: str) -> str:
    fname = chech_param(conf, param_name)
    fpath = os.path.join(parent_path, fname)
    if not os.path.exists(fpath):
        logger.warning('Missing [%s] filename in dir [%s]', fname, parent_path)
        exit(1)
    return fpath


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
    if hasattr(args, 'no_misc') and args.no_misc:
        del_misc = ['B-MISC', 'I-MISC']
    if not hasattr(args, 'pro') or not args.pro:
        del_misc = ['B-PRO', 'I-PRO']
    if not hasattr(args, 'evt') or not args.evt:
        del_misc = ['B-EVT', 'I-EVT']
    return del_misc
