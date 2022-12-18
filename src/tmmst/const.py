default_tmp_dir = 'tmp'
default_data_dir = 'data'
default_models_dir = 'models'
default_corpora_dir = 'corpora'

ner_tags = {
    'O': 0,
    'B-PER': 0, 'I-PER': 0,
    'B-LOC': 0, 'I-LOC': 0,
    'B-ORG': 0, 'I-ORG': 0,
    'B-MISC': 0, 'I-MISC': 0
}

ner_tags_list = [k for k, v in ner_tags.items()]

model_name_map = {
    'mcbert': 'bert-base-multilingual-cased',
    'xlmrb': 'xlm-roberta-base',
    'xlmrl': 'xlm-roberta-large'
}