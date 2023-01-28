#!/usr/bin/env python
import tmmst
import tmmst.data
import tmmst.const
import tmmst.args
import os
import argparse
import json
import re
import zipfile
import shutil
import logging
from typing import Dict, List, Any, Callable
from io import StringIO

logger = logging.getLogger('prep')
logger.addFilter(tmmst.fmt_filter)


def read_file_skip_lines(path: str, skip: int, line_clbk: Callable = None) -> str:
    file1 = open(path, 'r')
    buf = StringIO()
    count = 0
    while True:
        count += 1
        line = file1.readline()
        if not line:
            break
        if count <= skip:
            continue
        if line_clbk is not None:
            line_clbk(count - 1, line.strip())
        buf.write(line)
    return buf.getvalue()


def map_ner(map_filter: Dict, ner: str) -> str:
    mapped = map_filter.get(ner)
    if mapped is None:
        return ner
    else:
        return mapped


def add_tag_to_stats(stats: Dict, tag: str) -> None:
    ner_count = stats['tags'].get(tag)
    if ner_count is None:
        ner_count = 0
    stats['tags'][tag] = ner_count + 1
    if tag == 'O':
        return
    stats['tags']['NER'] = stats['tags']['NER'] + 1
    base_tag = tag.partition('-')[2]
    ner_count = stats['tags'].get(base_tag)
    if ner_count is None:
        ner_count = 0
    stats['tags'][base_tag] = ner_count + 1


def add_seq_to_stats(stats: Dict, seq: List, sentence_id: str, sentence: str) -> None:
    stats['num_sent'] = stats['num_sent'] + 1
    seq_len = len(seq)
    seq_len_cls: Dict = stats['seq_len_cls']
    if seq_len <= 32:
        seq_len_cls['32'] = seq_len_cls['32'] + 1
    elif seq_len <= 64:
        seq_len_cls['64'] = seq_len_cls['64'] + 1
    elif seq_len <= 128:
        seq_len_cls['128'] = seq_len_cls['128'] + 1
    elif seq_len <= 256:
        seq_len_cls['256'] = seq_len_cls['256'] + 1
        logger.info('Found sequence longer than 128 tokens! Manual check needed at sentence id [%s][%s]!',
                    sentence_id, sentence)
    elif seq_len <= 512:
        seq_len_cls['512'] = seq_len_cls['512'] + 1
        logger.info('Found sequence longer than 256 tokens! Manual check needed at sentence id [%s][%s]!',
                    sentence_id, sentence)
    else:
        logger.info('Found sequence longer than 512 tokens! Manual check needed at sentence id [%s][%s]!',
                    sentence_id, sentence)
    if seq_len > stats['longest_seq']:
        stats['longest_seq'] = seq_len


def parse_mwt_token(r: str) -> List[str]:
    mwt = r.split('-')
    result = []
    for x in range(int(mwt[0]), int(mwt[1]) + 1):
        result.append(str(x))
    return result


# noinspection DuplicatedCode
def conll2csv(conll_path: str, base_name: str, append: bool, ner_tag_idx: int, map_filter: Dict[str, Any] = None):
    if map_filter is None:
        map_filter = {}
    csv_fname = base_name + '.csv'
    json_fname = base_name + '.json'
    conll_fname = base_name + '.conll'
    csv = open(csv_fname, "a" if append else "w")
    csv.write('sentence,ner\n')
    logger.debug('Loading data [%s]', conll_path)
    logger.debug('Reformatting data NER [%s -> %s]...', conll_path, csv_fname)
    stats = {
        'tags': {
            'NER': 0, 'PER': 0, 'LOC': 0, 'ORG': 0, 'MISC': 0
        },
        'num_sent': 0,
        'longest_seq': 0,
        'seq_len_cls': {
            '32': 0, '64': 0, '128': 0, '256': 0, '512': 0
        }
    }
    stats['tags'].update(tmmst.const.ner_tags)
    sentence = {'id': None, 'tokens': [], 'text': ''}
    max_seq_len = map_filter.get('max_seq_len', 128)
    stop_at = map_filter.get('stop_at', -1)
    with open(conll_path) as fp:
        line = 'whatever'
        while line:
            line = fp.readline()
            if line.startswith('#'):
                # sent_id = train-s1
                if line.startswith("# sent_id = "):
                    sentence['id'] = line[12:].strip()
                # text = Proces privatizacije na Kosovu pod povećalom
                if line.startswith("# text = "):
                    sentence['text'] = line[9:].strip()
                continue
            if line == '\n' and sentence and sentence['tokens']:
                # process
                ner_tags = []
                sent_len = len(sentence['tokens'])
                if max_seq_len is not None and sent_len > max_seq_len:
                    logger.warning('Found sequence longer [%d] than [%d] tokens! Filtered out sentence id [%s][%s]!',
                                   sent_len, max_seq_len, sentence['id'], sentence['text'])
                    sentence = {'id': None, 'tokens': [], 'text': ''}
                    continue
                csv.write('"')
                mwt = []
                for token in sentence['tokens']:
                    if ner_tags:
                        csv.write(' ')
                    if token[1] == '"':
                        csv.write('""')
                    elif '"' in token[1]:
                        token[1].replace('"', '""')
                    else:
                        csv.write(token[1])
                    if "-" in token[0]:
                        mwt = parse_mwt_token(token[0])
                    if token[0] in mwt:
                        mwt.remove(token[0])
                        continue
                    ner_tag = token[ner_tag_idx]
                    if 'NER' in ner_tag:
                        ner_tag = re.findall(r'.*NER=([^|]+)', ner_tag)
                        if not ner_tag or len(ner_tag) <= 0:
                            logger.warning('Unable to parse NER tag at [%s:%s]', sentence['id'], sentence['text'])
                        ner_tag = ner_tag[0].strip()
                    if 'ner' in ner_tag:
                        ner_tag = re.findall(r'.*ner=([^|]+)', ner_tag)
                        if not ner_tag or len(ner_tag) <= 0:
                            logger.warning('Unable to parse ner tag at [%s:%s]', sentence['id'], sentence['text'])
                        ner_tag = ner_tag[0].strip()
                    if not ner_tag:
                        ner_tag = 'O'
                    ner_tag = ner_tag.strip()
                    ner_tag = map_ner(map_filter, ner_tag)
                    ner_tags.append(ner_tag)
                    add_tag_to_stats(stats, ner_tag)
                csv.write('",')
                csv.write(' '.join(ner_tags))
                csv.write("\n")
                add_seq_to_stats(stats, ner_tags, sentence['id'], sentence['text'])
                if stop_at > 0 and stats['num_sent'] >= stop_at:
                    logger.info("Forcefully stopping at sentence [%s]", stop_at)
                    break
                sentence = {'id': None, 'tokens': [], 'text': ''}
                continue
            data = line.split('\t')
            sentence['tokens'].append(data)

    logger.info('Reformatted data [%s -> %s]', conll_path, csv_fname)
    logger.info('Reformatting stats: %s', stats)
    with open(json_fname, 'w') as outfile:
        json.dump(stats, outfile, indent=2)
    csv.close()
    shutil.copyfile(conll_path, conll_fname)
    with zipfile.ZipFile(conll_fname + '.zip', 'w', compression=zipfile.ZIP_BZIP2, compresslevel=9) as myzip:
        myzip.write(conll_fname)
        myzip.close()


def bsnlp_init_anno_record(record: Dict, map_filter: Dict[str, Any]):
    ner_tags = []

    def process_ner_line(idx: int, line: str):
        data = line.split('\t')
        tag = data[2]
        form = data[0]
        tokens = map_filter['tokenizer'](form).sentences[0].tokens
        lower = True
        for t in tokens:
            if t.text.isalpha() and not t.text.islower():
                lower = False
                break

        ner_tags.append({
            'tag': tag,
            'form': form,
            'tokens': [x.text for x in tokens],
            'lower': lower,
            't_len': len(tokens),
            'sort': len(tokens) * 10000 + len(form)
        })

    read_file_skip_lines(record['a_fname'], 1, process_ner_line)
    ner_tags.sort(key=lambda x: x['sort'], reverse=True)
    record['a_ner_t'] = ner_tags


def bsnlp_process_raw_record(record: Dict, map_filter: Dict):
    record['r_text'] = read_file_skip_lines(record['r_fname'], 4)
    idx = 0
    text_token_list = []
    token_list = []
    doc = map_filter['tokenizer'](record['r_text'])
    lowerc = True
    for ner_tag in record['a_ner_t']:
        if not ner_tag['lower']:
            lowerc = False
            break

    for sent in doc.sentences:
        for sent_tok in sent.tokens:
            idx = record['r_text'].index(sent_tok.text, idx)
            sent_tok._start_char = idx
            sent_tok._ner = 'O'
            sent_tok._end_char = idx + len(sent_tok.text) - 1
            token_list.append(sent_tok)
            if lowerc:
                text_token_list.append(sent_tok.text.lower())
            else:
                text_token_list.append(sent_tok.text)
            idx = sent_tok.end_char + 1

    count = 0
    for ner_tag in record['a_ner_t']:
        ner_t_len = ner_tag['t_len']
        for idx in range(len(text_token_list) - ner_t_len + 1):
            if text_token_list[idx: idx + ner_t_len] == ner_tag['tokens']:
                for j in range(idx, idx + ner_t_len):
                    if token_list[j]._ner != 'O':
                        break
                    if j == idx:
                        token_list[j]._ner = 'B-' + ner_tag['tag']
                        count = + 1
                    else:
                        token_list[j]._ner = 'I-' + ner_tag['tag']
    if count == 0 and len(record['a_ner_t']) > 0:
        logger.warning('No NER matched in [%s] with annotations in [%s]!', record['r_fname'], record['a_fname'])
    record['conll'] = '# new_doc_id = ' + record['topic'] + '-' + record['id'] + '\n' + tmmst.data.to_conll(doc)


def bsnlp_create_records(bsnlp_path: str, lang: str, map_filter: Dict = None) -> Dict[str, Dict[str, Any]]:
    if map_filter is None:
        map_filter = {}
    ignore_dirs = map_filter.get('ignore_dirs')
    if ignore_dirs is None:
        ignore_dirs = []
    anno_path = os.path.join(bsnlp_path, 'annotated')
    raw_path = os.path.join(bsnlp_path, 'raw')
    anno_records = {}
    _, dirs, _ = next(x for x in os.walk(anno_path))
    dirs.sort()
    for d in dirs:
        if d in ignore_dirs:
            continue
        logger.debug('Processing directory [%s]', d)
        a_lang_path = os.path.join(anno_path, d, lang)
        r_lang_path = os.path.join(raw_path, d, lang)
        if os.path.exists(a_lang_path) and os.path.exists(r_lang_path):
            _, _, a_files = next(x for x in os.walk(a_lang_path))
            _, _, r_files = next(x for x in os.walk(r_lang_path))
            a_files.sort()
            r_files.sort()
            for af in a_files:
                afnum = re.findall(r'(\d+([-_]\d+)*)', af)
                if not afnum or len(afnum) <= 0 or len(afnum[0]) <= 0:
                    logger.warning('Unable to parse id from annotated file [%s]', af)
                    continue
                a_fname = os.path.join(a_lang_path, af)
                record = {
                    'a_fname': a_fname,
                    'topic': d,
                    'id': afnum[0][0]
                }
                bsnlp_init_anno_record(record, map_filter)
                anno_records[d + '-' + afnum[0][0]] = record
            for rf in r_files:
                # rfnum = re.findall(r'\d+', rf)
                rfnum = re.findall(r'(\d+([-_]\d+)*)', rf)
                if not rfnum or len(rfnum) <= 0 or len(rfnum[0]) <= 0:
                    logger.warning('Unable to parse id from raw file [%s]', rf)
                    continue
                record = anno_records.get(d + '-' + rfnum[0][0])
                if record is None:
                    logger.warning('Unable to find matching annotated file [%s]', rf)
                    continue
                record['r_fname'] = os.path.join(r_lang_path, rf)
                bsnlp_process_raw_record(record, map_filter)

    return anno_records


def bsnlp2csv(bsnlp_path: str, base_name: str, append: bool, map_filter: Dict = None):
    if map_filter is None:
        map_filter = {}
    lang = map_filter.get('lang')
    if lang is None:
        raise ValueError('Missing map_filter lang parameter!')
    conll_fname = os.path.join('tmp', 'out.conll')

    anno_records = bsnlp_create_records(bsnlp_path, lang, map_filter)
    conll = open(conll_fname, "a" if append else "w")
    for k, record in anno_records.items():
        conll.write(record['conll'])
    conll.close()
    logger.info('Reformatted data [%s -> %s]', bsnlp_path, conll_fname)
    conll2csv(conll_fname, base_name, append, 9, map_filter)


def filter_wikiann(lang: str, proc_fname: str) -> None:
    data = ''
    invalid = {"''", "'", "]]", "[[", "==", "**", "``"}
    counter = 0
    sent_id = 0
    prefix = lang + ':'
    with open(os.path.join(proc_fname), 'rt', encoding='utf-8') as fp:
        line = 'whatever'
        while line:
            line = fp.readline()
            if line == '\n':
                if counter > 0:
                    data += '\n'
                counter = 0
                continue
            if line == '':
                continue
            tokens = line.split('\t')
            if tokens[0] and tokens[0].startswith(prefix):
                tokens[0] = tokens[0][len(prefix):]
            if tokens[0] in invalid:
                continue
            if tokens[0].startswith("''"):
                tokens[0] = tokens[0][2:]
            if tokens[0].startswith("**"):
                tokens[0] = tokens[0][2:]
            if counter == 0:
                data += '# sent_id = ' + str(sent_id) + '\n'
                sent_id += 1
            if counter == 0 and tokens[0] == '-':
                continue
            if counter == 0 and tokens[0] == '–':
                continue
            if counter == 0 and tokens[0] == ',':
                continue
            if counter == 0 and tokens[0] == ')':
                continue
            data += str(counter) + '\t' + tokens[0] + '\t' + tokens[1]
            counter += 1

    with open(os.path.join(proc_fname), 'wt', encoding='utf-8') as fp:
        fp.write(data)


def filter_cnec(proc_fname: str):
    # I = ORG
    # T = MISC (time) -> O
    # P = PER
    # G = LOC
    # M = MISC (media) -> ORG (if not @)
    # O = MISC (product)
    # A = MISC (address, number) -> O
    # B-[^ITPGMAO]{1,}$
    data = ''
    counter = 0
    sent_id = 0
    with open(os.path.join(proc_fname), 'rt', encoding='utf-8') as fp:
        line = 'whatever'
        while line:
            line = fp.readline()
            if line == '\n':
                if counter > 0:
                    data += '\n'
                counter = 0
                continue
            if line == '':
                continue
            tokens = line.split('\t')
            if counter == 0:
                data += '# sent_id = ' + str(sent_id) + '\n'
                sent_id += 1
            if '@' in tokens[0] and tokens[3].endswith('M'):
                tokens[3] = 'O'
            data += str(counter) + '\t' + tokens[0] + '\t' + tokens[3]
            counter += 1

    with open(os.path.join(proc_fname), 'wt', encoding='utf-8') as fp:
        fp.write(data)
    pass


def prep_data(args, confs: List[Dict]) -> None:
    for conf in confs:
        type = conf.get('type')
        if type == 'wikiann':
            ner_conll_idx = tmmst.args.chech_param(conf, 'ner_conll_idx')
            target_base_name = tmmst.args.chech_param(conf, 'result_name')
            map_filter = tmmst.args.chech_param(conf, 'map_filter')
            lang = tmmst.args.chech_param(conf, 'lang')
            zip_fname = tmmst.args.chech_dir_param(conf, 'zip', args.corpora_dir)
            zip_dir = os.path.join(tmmst.const.default_tmp_dir, target_base_name)
            if not os.path.exists(zip_dir):
                os.mkdir(zip_dir)
            zipfile.ZipFile(zip_fname).extractall(zip_dir)
            proc_fname = tmmst.args.chech_dir_param(conf, 'proc_file', tmmst.const.default_tmp_dir)
            with open(os.path.join(proc_fname, 'train'), 'rt', encoding='utf-8') as fp:
                data = fp.read()
            with open(os.path.join(proc_fname, 'dev'), 'rt', encoding='utf-8') as fp:
                data += "\n"
                data += fp.read()
            with open(os.path.join(proc_fname, 'test'), 'rt', encoding='utf-8') as fp:
                data += "\n"
                data += fp.read()
            with open(os.path.join(proc_fname, 'extra'), 'rt', encoding='utf-8') as fp:
                data += "\n"
                data += fp.read()
            proc_fname = os.path.join(proc_fname, lang + '-wann.conll')
            with open(proc_fname, 'wt', encoding='utf-8') as fp:
                fp.write(data)
            filter_wikiann(lang, proc_fname)
            conll2csv(proc_fname, os.path.join(args.data_dir, target_base_name), False, ner_conll_idx, map_filter)
        if type == 'cnec':
            ner_conll_idx = tmmst.args.chech_param(conf, 'ner_conll_idx')
            target_base_name = tmmst.args.chech_param(conf, 'result_name')
            map_filter = tmmst.args.chech_param(conf, 'map_filter')
            zip_fname = tmmst.args.chech_dir_param(conf, 'zip', args.corpora_dir)
            zipfile.ZipFile(zip_fname).extractall(tmmst.const.default_tmp_dir)
            proc_fname = tmmst.args.chech_dir_param(conf, 'proc_file', tmmst.const.default_tmp_dir)
            with open(os.path.join(proc_fname, 'dtest.conll'), 'rt', encoding='utf-8') as fp:
                data = fp.read()
            with open(os.path.join(proc_fname, 'train.conll'), 'rt', encoding='utf-8') as fp:
                data += "\n"
                data += fp.read()
            with open(os.path.join(proc_fname, 'etest.conll'), 'rt', encoding='utf-8') as fp:
                data += "\n"
                data += fp.read()
            proc_fname = os.path.join(proc_fname, 'cs-cnec.conll')
            with open(proc_fname, 'wt', encoding='utf-8') as fp:
                fp.write(data)
            filter_cnec(proc_fname)
            conll2csv(proc_fname, os.path.join(args.data_dir, target_base_name), False, ner_conll_idx, map_filter)
        if type == 'conll':
            zip_fname = tmmst.args.chech_dir_param(conf, 'zip', args.corpora_dir)
            zipfile.ZipFile(zip_fname).extractall(tmmst.const.default_tmp_dir)
            proc_fname = tmmst.args.chech_dir_param(conf, 'proc_file', tmmst.const.default_tmp_dir)
            target_base_name = tmmst.args.chech_param(conf, 'result_name')
            map_filter = tmmst.args.chech_param(conf, 'map_filter')
            logger.debug('Converting conll data [%s -> %s]...', proc_fname, target_base_name)
            ner_conll_idx = tmmst.args.chech_param(conf, 'ner_conll_idx')
            conll2csv(proc_fname, os.path.join(args.data_dir, target_base_name), False, ner_conll_idx, map_filter)
            logger.info('Converted data [%s -> %s]', proc_fname, target_base_name)
        if type == 'bsnlp':
            zip_fname = tmmst.args.chech_dir_param(conf, 'zip', args.corpora_dir)
            zipfile.ZipFile(zip_fname).extractall(tmmst.const.default_tmp_dir)
            proc_fname = tmmst.args.chech_dir_param(conf, 'proc_file', tmmst.const.default_tmp_dir)
            target_base_name = tmmst.args.chech_param(conf, 'result_name')
            map_filter = tmmst.args.chech_param(conf, 'map_filter')
            logger.debug('Converting BSNLP data [%s -> %s]...', proc_fname, target_base_name)
            bsnlp2csv(proc_fname, os.path.join(args.data_dir, target_base_name), False, map_filter)
            logger.info('Converted data [%s -> %s]', proc_fname, target_base_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NER Data preparation and normalization for Slovene, Croatian and Serbian language')
    parser.add_argument('lang', help='language of the text',
                        choices=['sl', 'hr', 'sr', 'bs', 'mk', 'sq', 'cs'])
    parser.add_argument(
        '-d', '--data_dir', help='Data output directory', type=tmmst.args.dir_path,
        default=tmmst.const.default_data_dir)
    parser.add_argument(
        '-c', '--corpora_dir', help='Corpora input directory', type=tmmst.args.dir_path,
        default=tmmst.const.default_corpora_dir)
    parser.add_argument(
        '-s', '--data_split',
        help='Data split in % separated with colon: '
             'For example "80:10" would produce 80% train, 10% evaluation and 10% test data set size. ',
        type=str, default='80:10'
    )
    parser.add_argument(
        '-r', '--non_reproducible_shuffle', help='Non reproducible data shuffle.', action='store_true', default=False
    )
    args = parser.parse_args()
    if args.lang == 'sl':
        tokenizer = tmmst.data.get_classla_tokenizer(args.lang)
        confs = [
            {
                'type': 'conll',
                'zip': 'SUK.CoNLL-U.zip',
                'proc_file': os.path.join('SUK.CoNLL-U', 'ssj500k-syn.ud.conllu'),
                'result_name': 'sl_500k',
                'ner_conll_idx': 9,
                'map_filter': {
                    'max_seq_len': 128,
                    'stop_at': 9483,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            },
            {
                'type': 'conll',
                'zip': 'SUK.CoNLL-U.zip',
                'proc_file': os.path.join('SUK.CoNLL-U', 'senticoref.ud.conllu'),
                'result_name': 'sl_scr',
                'ner_conll_idx': 9,
                'map_filter': {
                    'max_seq_len': 128,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            },
            {
                'type': 'conll',
                'zip': 'SUK.CoNLL-U.zip',
                'proc_file': os.path.join('SUK.CoNLL-U', 'elexiswsd.ud.conllu'),
                'result_name': 'sl_ewsd',
                'ner_conll_idx': 9,
                'map_filter': {
                    'max_seq_len': 128,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            },
            {
                'type': 'bsnlp',
                'zip': 'bsnlp-2017-21.zip',
                'proc_file': 'bsnlp',
                'result_name': 'sl_bsnlp',
                'map_filter': {
                    'max_seq_len': 128,
                    'lang': 'sl',
                    'tokenizer': tokenizer,
                    'B-EVT': 'B-MISC', 'I-EVT': 'I-MISC',
                    'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC'
                }
            }
        ]
        prep_data(args, confs)
        tmmst.data.split_data(args, confs)
    if args.lang == 'hr':
        tokenizer = tmmst.data.get_classla_tokenizer(args.lang)
        confs = [
            {
                'type': 'conll',
                'zip': 'hr500k-1.0.zip',
                'proc_file': os.path.join('hr500k.conll', 'hr500k.conll'),
                'result_name': 'hr_500k',
                'ner_conll_idx': 10,
                'map_filter': {
                    'max_seq_len': 128,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            },
            {
                'type': 'bsnlp',
                'zip': 'bsnlp-2017-21.zip',
                'proc_file': 'bsnlp',
                'result_name': 'hr_bsnlp',
                'map_filter': {
                    'max_seq_len': 128,
                    'lang': 'hr',
                    'tokenizer': tokenizer,
                    'B-EVT': 'B-MISC', 'I-EVT': 'I-MISC',
                    'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC'
                }
            }
        ]
        prep_data(args, confs)
        tmmst.data.split_data(args, confs)
    if args.lang == 'sr':
        confs = [
            {
                'type': 'conll',
                'zip': 'setimes-sr.conll.zip',
                'proc_file': os.path.join('setimes-sr.conll', 'set.sr.conll'),
                'result_name': 'sr_set',
                'ner_conll_idx': 10,
                'map_filter': {
                    'max_seq_len': 128,
                    'B-loc': 'B-LOC', 'I-loc': 'I-LOC',
                    'B-org': 'B-ORG', 'I-org': 'I-ORG',
                    'B-per': 'B-PER', 'I-per': 'I-PER',
                    'B-misc': 'B-MISC', 'I-misc': 'I-MISC',
                    'B-deriv-per': 'B-PER', 'I-deriv-per': 'I-PER'
                }
            }
        ]
        prep_data(args, confs)
        tmmst.data.split_data(args, confs)
    if args.lang == 'cs':
        confs = [
            {
                'type': 'cnec',
                'zip': 'CNEC_2.0_konkol.zip',
                'proc_file': 'CNEC_2.0_konkol',
                'result_name': 'cs_cnec',
                'ner_conll_idx': 2,
                'map_filter': {
                    'max_seq_len': 128,
                    'B-G': 'B-LOC', 'I-G': 'I-LOC',
                    'B-I': 'B-ORG', 'I-I': 'I-ORG',
                    'B-M': 'B-ORG', 'I-M': 'I-ORG',
                    'B-P': 'B-PER', 'I-P': 'I-PER',
                    'B-O': 'B-MISC', 'I-O': 'I-MISC',
                    'B-T': 'O', 'I-T': 'O',
                    'B-A': 'O', 'I-A': 'O'
                }
            }
        ]
        prep_data(args, confs)
        tmmst.data.split_data(args, confs)
    if args.lang == 'bs':
        confs = [
            {
                'type': 'wikiann',
                'lang': 'bs',
                'zip': 'bs-wann.zip',
                'proc_file': 'bs_wann',
                'result_name': 'bs_wann',
                'ner_conll_idx': 2,
                'map_filter': {
                    'max_seq_len': 128
                }
            }
        ]
        prep_data(args, confs)
        tmmst.data.split_data(args, confs)
    if args.lang == 'mk':
        confs = [
            {
                'type': 'wikiann',
                'lang': 'mk',
                'zip': 'mk-wann.zip',
                'proc_file': 'mk_wann',
                'result_name': 'mk_wann',
                'ner_conll_idx': 2,
                'map_filter': {
                    'max_seq_len': 128
                }
            }
        ]
        prep_data(args, confs)
        tmmst.data.split_data(args, confs)
    if args.lang == 'sq':
        confs = [
            {
                'type': 'wikiann',
                'lang': 'sq',
                'zip': 'sq-wann.zip',
                'proc_file': 'sq_wann',
                'result_name': 'sq_wann',
                'ner_conll_idx': 2,
                'map_filter': {
                    'max_seq_len': 128
                }
            }
        ]
        prep_data(args, confs)
        tmmst.data.split_data(args, confs)
