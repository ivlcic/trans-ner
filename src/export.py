#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import tmmst
import tmmst.args
import tmmst.const
import argparse
import collections
import logging
import os


from tmmst.torch import TrainedModelContainer

logger = logging.getLogger('export')
logger.addFilter(tmmst.fmt_filter)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ArgNamespace = collections.namedtuple(
    'ArgNamespace', [
        'pretrained_model', 'data_dir', 'models_dir', 'target_model', "limit_cuda_device"
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Helper script to export model checkpoint to a directory.')
    parser.add_argument('pretrained_model', help='Pretrained model to export.')
    parser.add_argument('target_model', help='Target model (directory) name to export to.')
    tmmst.args.add_common_dirs(parser)
    tmmst.args.add_common_arguments(parser)
    # noinspection PyTypeChecker
    args: ArgNamespace = parser.parse_args()

    if args.limit_cuda_device is not None:
        logger.info("Will run on specified cuda [%s] device only!", args.limit_cuda_device)

    mc = TrainedModelContainer(
        tmmst.args.get_pretrained_model_path(args), tmmst.args.get_tags_to_remove(args)
    )

    if os.sep in args.target_model:
        export_path = os.path.join(args.target_model)
    else:
        export_path = os.path.join(args.models_dir, args.target_model)

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    logger.info("Exporting model and tokenizer:")
    logger.info("%s", mc.model.save_pretrained(export_path))
    logger.info("%s", mc.tokenizer.save_pretrained(export_path))
