import json
import logging
import os
import classla

import tmmst.const
import tmmst.args
from typing import List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger('data')
logger.addFilter(tmmst.fmt_filter)


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_classla_tokenizer(lang: str):
    classla_dir = os.path.join(tmmst.const.default_tmp_dir, 'classla')
    if not os.path.exists(classla_dir):
        os.makedirs(classla_dir)
    if not os.path.exists(os.path.join(classla_dir, lang)):
        classla.download(lang, classla_dir)
    return classla.Pipeline(lang, dir=classla_dir, processors="tokenize", download_method=2)


def split_data(args, confs: List[Dict]) -> None:
    ds = [int(x) for x in args.data_split.split(':')]
    if len(ds) != 2:
        raise ValueError("We need two-way split arg: 'train:eval' we use remains as 'test' "
                         "i.e. split arg '80:15' would leave us with 5% test set size")
    if sum(ds) >= 100:
        raise ValueError("Data split sum must be less than 100 since we also need a test set!")

    training_sets: List[pd.DataFrame] = []
    evaluation_sets: List[pd.DataFrame] = []
    test_sets: List[pd.DataFrame] = []
    for conf in confs:
        target_base_name = tmmst.args.chech_param(conf, 'result_name')
        data = pd.read_csv(os.path.join(args.data_dir, target_base_name + '.csv'))

        # Shuffle the whole dataset first
        if args.non_reproducible_shuffle:
            data = data.sample(frac=1).reset_index()
            logger.info("Done non-reproducible data shuffle.")
        else:
            data = data.sample(frac=1, random_state=2611).reset_index()
            logger.info("Done reproducible data shuffle.")

        data_len = len(data)
        train_n = int((ds[0] / 100) * data_len)
        eval_n = train_n + int((ds[1] / 100) * data_len)
        test_n = data_len - eval_n
        logger.info("Splitting the data [%s:%s] proportionally to [%s] => [train:%s,eval:%s,test:%s]",
                    target_base_name, data_len, args.data_split, train_n, eval_n, test_n)
        training_data, evaluation_data, test_data = np.split(data, [train_n, eval_n])
        training_sets.append(training_data)
        training_data.to_csv(
            os.path.join(args.data_dir, target_base_name + '.train.csv'), index=False, encoding='utf-8'
        )
        evaluation_sets.append(evaluation_data)
        evaluation_data.to_csv(
            os.path.join(args.data_dir, target_base_name + '.eval.csv'), index=False, encoding='utf-8'
        )
        test_sets.append(test_data)
        test_data.to_csv(
            os.path.join(args.data_dir, target_base_name + '.test.csv'), index=False, encoding='utf-8'
        )

    pd.concat(training_sets).to_csv(
        os.path.join(args.data_dir, args.lang + '.train.csv'), index=False, encoding='utf-8'
    )
    pd.concat(evaluation_sets).to_csv(
        os.path.join(args.data_dir, args.lang + '.eval.csv'), index=False, encoding='utf-8'
    )
    pd.concat(test_sets).to_csv(
        os.path.join(args.data_dir, args.lang + '.test.csv'), index=False, encoding='utf-8'
    )


def load_corpus(path_prefixes: List[str]) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

    training_sets: List[pd.DataFrame] = []
    evaluation_sets: List[pd.DataFrame] = []
    test_sets: List[pd.DataFrame] = []
    for path_prefix in path_prefixes:
        logger.debug("Loading corpus [%s]...", path_prefix)
        training_sets.append(
            pd.read_csv(path_prefix + '.train.csv')
        )
        evaluation_sets.append(
            pd.read_csv(path_prefix + '.eval.csv')
        )
        test_sets.append(
            pd.read_csv(path_prefix + '.test.csv')
        )
        logger.debug("Loaded corpus [%s]", path_prefix)

    return pd.concat(training_sets), pd.concat(evaluation_sets), pd.concat(test_sets)
