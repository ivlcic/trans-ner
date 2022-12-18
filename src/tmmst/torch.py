import tmmst
import logging
import pandas as pd
import numpy as np
import torch
import evaluate


from typing import List
from tmmst.const import ner_tags as const_ner_tags
from tokenizers.tokenizers import Encoding
from transformers import AutoTokenizer, AutoModelForTokenClassification, BatchEncoding
from torch.utils.data import DataLoader

logger = logging.getLogger('train')
logger.addFilter(tmmst.fmt_filter)


class ModelContainer(torch.nn.Module):

    def __init__(self, model_dir: str, model_name: str, remove_ner_tags=None):
        super(ModelContainer, self).__init__()
        if remove_ner_tags is None:
            remove_ner_tags = []
        self.model = None
        self.tokenizer = None
        self.remove_ner_tags = remove_ner_tags
        label_id_map = {k: v for v, k in enumerate(const_ner_tags) if k not in self.remove_ner_tags}
        self.label_id_map = label_id_map
        ids_to_labels = {v: k for v, k in enumerate(const_ner_tags) if k not in self.remove_ner_tags}
        self.ids_to_labels = ids_to_labels
        self.metric = evaluate.load("seqeval")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, cache_dir=model_dir, num_labels=len(label_id_map),
            id2label=ids_to_labels, label2id=label_id_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=model_dir
        )
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(device)
        self.device = device

    def forward(self, input_id, mask, label):
        output = self.model(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output

    def compute_metrics(self, p, test: bool = False):
        predictions_list, labels_list = p

        # select predicted index with maximum logit for each token
        predictions_list = np.argmax(predictions_list, axis=2)

        tagged_predictions_list = []
        tagged_labels_list = []
        for predictions, labels in zip(predictions_list, labels_list):
            tagged_predictions = []
            tagged_labels = []
            for pid, lid in zip(predictions, labels):
                if lid != -100:
                    tagged_predictions.append(self.ids_to_labels[pid])
                    tagged_labels.append(self.ids_to_labels[lid])
            tagged_predictions_list.append(tagged_predictions)
            tagged_labels_list.append(tagged_labels)

        results = self.metric.compute(
            predictions=tagged_predictions_list, references=tagged_labels_list, scheme='IOB2', mode='strict'
        )
        if test:
            return results
        logger.info("Batch eval: %s", results)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


class TrainedModelContainer(ModelContainer):
    def __init__(self, model_dir: str, remove_ner_tags=None):
        super().__init__(None, model_dir, remove_ner_tags)


class DataSequence(torch.utils.data.Dataset):

    def align_labels(self, encoded: Encoding, labels: List[str]):
        word_ids = encoded.word_ids
        label_ids = []
        max_idx = len(labels)
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx < 0 or word_idx >= max_idx:
                label_ids.append(-100)
            else:
                label_ids.append(self.label_id_map[labels[word_idx]])
        return label_ids

    def del_ner(self, ner_text: str):
        for rem_ner_tag in self.remove_ner_tags:
            ner_text = ner_text.replace(rem_ner_tag, 'O')
        return ner_text

    def __init__(self, model: ModelContainer, data: pd.DataFrame, max_seq_len: int):
        self.remove_ner_tags = model.remove_ner_tags
        self.label_id_map = model.label_id_map
        self.max_seq_len = max_seq_len

        ner_labels = [self.del_ner(i).split() for i in data['ner'].values.tolist()]

        unique_labels = set()
        for lb in ner_labels:
            [unique_labels.add(i) for i in lb if i not in unique_labels]
        if unique_labels != self.label_id_map.keys():
            logger.error("Unexpected NER tag [%s] in [%s] in dataset!",
                         unique_labels, self.label_id_map.keys())
            exit(1)

        sentences = data['sentence'].values.tolist()
        self.encodings: BatchEncoding = model.tokenizer(
            sentences, padding='max_length', max_length=max_seq_len, truncation=True, return_tensors="pt"
        )
        self.labels = []
        self.ner_tags = ner_labels
        for i, e in enumerate(self.encodings.encodings):
            self.labels.append(self.align_labels(e, ner_labels[i]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item