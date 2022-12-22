#!/usr/bin/env python
import tmmst
import tmmst.data
import tmmst.const
import tmmst.args
import os
import logging
import argparse
import pandas as pd
import json
import collections

from tmmst.torch import DataSequence, ModelContainer
from transformers import TrainingArguments, Trainer


logger = logging.getLogger('train')
logger.addFilter(tmmst.fmt_filter)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ArgNamespace = collections.namedtuple(
    'ArgNamespace', [
        'lang', 'corpora', 'pretrained_model', 'data_dir', 'models_dir', 'learn_rate', 'epochs',
        'data_split', 'non_reproducible_shuffle', 'batch', 'max_seq_len', 'no_misc', "limit_cuda_device",
        "target_model_name"
    ]
)


def train(args: ArgNamespace, mc: tmmst.torch.ModelContainer, training_args: TrainingArguments,
          train_data: pd.DataFrame, eval_data: pd.DataFrame, test_data: pd.DataFrame) -> None:

    logger.debug("Constructing train data set [%s]...", len(train_data))
    train_set = DataSequence(mc, train_data, args.max_seq_len)
    logger.info("Constructed train data set [%s].", len(train_data))
    logger.debug("Constructing evaluation data set [%s]...", len(eval_data))
    eval_set = DataSequence(mc, eval_data, args.max_seq_len)
    logger.info("Constructed evaluation data set [%s].", len(eval_data))
    logger.debug("Constructing test data set [%s]...", len(test_data))
    test_set = DataSequence(mc, test_data, args.max_seq_len)
    logger.info("Constructed test data set [%s].", len(test_data))

    training_args.logging_steps = len(train_set)

    trainer = Trainer(
        model=mc.model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=mc.tokenizer,
        compute_metrics=mc.compute_metrics
    )
    logger.debug("Starting training...")
    trainer.train()
    logger.info("Training done.")
    logger.debug("Starting evaluation...")
    trainer.evaluate()
    logger.info("Evaluation done.")

    logger.info("Starting test set evaluation...")
    predictions, labels, _ = trainer.predict(test_set)
    results = mc.compute_metrics((predictions, labels), True)
    logger.info("Test set evaluation results:")
    logger.info("%s", results)
    combined_results = {}
    if os.path.exists(os.path.join(args.models_dir, 'results_all.json')):
        with open(os.path.join(args.models_dir, 'results_all.json')) as json_file:
            combined_results = json.load(json_file)
    combined_results[args.target_model_name] = results
    with open(os.path.join(args.models_dir, 'results_all.json'), 'wt', encoding='utf-8') as fp:
        json.dump(combined_results, fp, cls=tmmst.data.NpEncoder)
    with open(os.path.join(args.models_dir, args.target_model_name + ".json"), 'wt') as fp:
        json.dump(results, fp, cls=tmmst.data.NpEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NER Neural train for Slovene, Croatian and Serbian language')
    parser.add_argument('corpora', help='Corpora to use', nargs='+',
                        choices=[
                            'sl_500k', 'sl_bsnlp', 'sl_ewsd', 'sl_scr', 'sl',
                            'hr_500k', 'hr_bsnlp', 'hr',
                            'sr_set', 'sr',
                            'bs_wann', 'bs',
                            'mk_wann', 'mk'
                        ])
    parser.add_argument('pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])
    tmmst.args.add_common_dirs(parser)
    parser.add_argument(
        '-l', '--learn_rate', help='Learning rate', type=float, default=5e-5
    )
    parser.add_argument(
        '-e', '--epochs', help='Number of epochs.', type=int, default=20
    )
    tmmst.args.add_common_arguments(parser)
    # noinspection PyTypeChecker
    args: ArgNamespace = parser.parse_args()

    if args.limit_cuda_device is not None:
        logger.info("Will run on specified cuda [%s] device only!", args.limit_cuda_device)

    model_name = args.pretrained_model + '-' + '.'.join(args.corpora)
    if args.no_misc:
        model_name += '-nomisc'
    args.target_model_name = model_name

    model_result_dir = os.path.join(args.models_dir, args.target_model_name)
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)

    training_args = TrainingArguments(
        output_dir=model_result_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        evaluation_strategy="epoch",
        #disable_tqdm=True,
        load_best_model_at_end=True,
        save_strategy='epoch',
        optim='adamw_torch',
        save_total_limit=1,
        metric_for_best_model='f1',
        logging_strategy='epoch',
    )

    mc = ModelContainer(
        tmmst.args.get_pretrained_model_path(args, True),
        tmmst.const.model_name_map[args.pretrained_model],
        tmmst.args.get_tags_to_remove(args)
    )
    path_prefix = []
    for corpus in args.corpora:
        path_prefix.append(os.path.join(args.data_dir, corpus))
    train_data, eval_data, test_data = tmmst.data.load_corpus(logger, path_prefix)
    train(args, mc, training_args, train_data, eval_data, test_data)
