#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import tmmst
import tmmst.data
import tmmst.args
import tmmst.const
import argparse
import collections
import logging
import os


from tmmst.torch import DataSequence, TrainedModelContainer
from transformers import TrainingArguments, Trainer

logger = logging.getLogger('test')
logger.addFilter(tmmst.fmt_filter)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ArgNamespace = collections.namedtuple(
    'ArgNamespace', [
        'lang', 'corpora', 'pretrained_model', 'data_dir', 'models_dir',
        'data_split', 'non_reproducible_shuffle', 'batch', 'max_seq_len', 'no_misc', "limit_cuda_device"
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NER Test')
    parser.add_argument('corpora', help='Corpora to use for testing', nargs='+',
                        choices=[
                            'sl_500k', 'sl_bsnlp', 'sl_ewsd', 'sl_scr', 'sl',
                            'hr_500k', 'hr_bsnlp', 'hr',
                            'sr_set', 'sr',
                            'bs_wann', 'bs',
                            'mk_wann', 'mk',
                            'sq_wann', 'sq',
                            'cs_bsnlp', 'cs_cnec', 'cs'
                            'bg_bsnlp', 'bg',
                            'uk_bsnlp', 'uk',
                            'ru_bsnlp', 'ru',
                            'pl_bsnlp', 'pl'
                        ])
    parser.add_argument('pretrained_model', help='Pretrained model to use for testing')
    tmmst.args.add_common_dirs(parser)
    tmmst.args.add_common_arguments(parser)
    # noinspection PyTypeChecker
    args: ArgNamespace = parser.parse_args()

    if args.limit_cuda_device is not None:
        logger.info("Will run on specified cuda [%s] device only!", args.limit_cuda_device)
    logger.info("Loading model from [%s] with disabled NER tags [%s]",
                tmmst.args.get_pretrained_model_path(args), tmmst.args.get_tags_to_remove(args))
    mc = TrainedModelContainer(
        tmmst.args.get_pretrained_model_path(args), tmmst.args.get_tags_to_remove(args)
    )

    training_args = TrainingArguments(
        args.models_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
    )

    trainer = Trainer(
        model=mc.model,
        args=training_args,
        tokenizer=mc.tokenizer,
        compute_metrics=mc.compute_metrics
    )
    logger.info("Starting test set evaluation...")

    path_prefix = []
    for corpus in args.corpora:
        path_prefix.append(os.path.join(args.data_dir, corpus))

    _, _, test_data = tmmst.data.load_corpus(path_prefix)
    test_set = DataSequence(mc, test_data, args.max_seq_len)
    predictions, labels, _ = trainer.predict(test_set)
    results = mc.compute_metrics((predictions, labels), True)
    logger.info("Test set evaluation results:")
    logger.info("%s", results)
