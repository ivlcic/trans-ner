#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import tmmst
import tmmst.args
import tmmst.data
import torch
import collections
import argparse
import logging

from typing import Dict
from tmmst.torch import TrainedModelContainer

logger = logging.getLogger('infer')
logger.addFilter(tmmst.fmt_filter)

ArgNamespace = collections.namedtuple(
    'ArgNamespace', [
        'lang', 'pretrained_model', 'data_dir', 'models_dir', 'no_misc', 'text'
    ]
)


def _tokenize(mc: TrainedModelContainer, word_list):
    # Lightly adapted from the pipeline._preprocess method
    model_inputs = mc.tokenizer(
        word_list,
        return_tensors="pt",
        truncation=False,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
        is_split_into_words=True,
    )
    if len(model_inputs["input_ids"][0]) > mc.tokenizer.model_max_length:
        sent = " ".join(word_list)
        logger.warning(f"Truncated long input sentence:\n{sent}")
        model_inputs = mc.tokenizer(
            word_list,
            return_tensors="pt",
            truncation=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            is_split_into_words=True,
        )

    model_inputs.to(mc.device)
    return model_inputs


def _infer(mc: TrainedModelContainer, model_inputs):
    # Lightly adapted from the pipeline._forward method
    special_tokens_mask = model_inputs.pop("special_tokens_mask")
    offset_mapping = model_inputs.pop("offset_mapping", None)
    with torch.no_grad():
        logits = mc.model(**model_inputs)[0]
    return {
        "logits": logits,
        "special_tokens_mask": special_tokens_mask,
        "offset_mapping": offset_mapping
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simple NER Neural inference script for manual checking.')
    parser.add_argument('pretrained_model', help='Pretrained model to use for inference')
    parser.add_argument('lang', help='language of the text needed for the word tokenizer.',
                        choices=['sl', 'hr', 'sr', 'bs', 'mk', 'sq', 'cs', 'pl', 'ru'])
    parser.add_argument('text', help='Text to classify')
    tmmst.args.add_common_dirs(parser)
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

    # noinspection PyTypeChecker
    args: ArgNamespace = parser.parse_args()

    mc = TrainedModelContainer(
        tmmst.args.get_pretrained_model_path(args), tmmst.args.get_tags_to_remove(args)
    )
    mc.model.eval()
    tokenizer = tmmst.data.get_classla_tokenizer(args.lang) \
        if args.lang in ['bg', 'hr', 'sl', 'sr', 'mk'] \
        else tmmst.data.get_stanza_tokenizer(args.lang)
    doc = tokenizer.process(args.text)
    for sent_idx, sentence in enumerate(doc.sentences):
        word_list = [v.text for v in sentence.tokens]
        model_inputs = _tokenize(mc, word_list)
        word_ids = model_inputs.word_ids()
        model_outputs = _infer(mc, model_inputs)
        logits = model_outputs["logits"][0]
        # scores = logits.softmax(1).max(axis=1).values.numpy().tolist()
        predicted_classes = logits.argmax(axis=1).cpu().numpy().tolist()

        for ix in range(1, len(model_inputs[0]) - 1):
            contd_cls_name = mc.ids_to_labels.get(predicted_classes[ix], 'O')
            token_idx = word_ids[ix]
            if token_idx >= len(sentence.tokens):
                continue
            token: Dict[str, any] = sentence.tokens[token_idx]
            if not hasattr(token, 'ner'):
                setattr(token, 'ner', contd_cls_name)
            else:
                token.ner = contd_cls_name
        sent_text = ''
        prev_token = None
        for v in sentence.tokens:
            if v.ner == 'O':
                if prev_token and prev_token.ner and prev_token.ner != 'O':
                    sent_text += ']-' + prev_token.ner[2:]
                if sent_text:
                    sent_text += ' '
                sent_text += v.text
                prev_token = v
                continue
            elif v.ner.startswith('B-'):
                sent_text += '[' + v.text
            else:
                sent_text += ' ' + v.text
            prev_token = v

        logger.debug('%s', sent_text)
