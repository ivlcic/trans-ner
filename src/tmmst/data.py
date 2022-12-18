import json
import logging
import os
import classla

import tmmst.const
from typing import List

import numpy as np
import pandas as pd


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


def load_corpus(logger: logging.Logger, path_prefixes: List[str]) \
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
